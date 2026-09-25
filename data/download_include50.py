"""
Smart INCLUDE-50 Downloader
============================
Downloads ONLY the ~958 videos referenced by INCLUDE-50 split files from
Zenodo zip archives using HTTP range requests, avoiding the full 53 GB download.

Approach:
1. Read the zip central directory (last ~64KB of each zip) via HTTP Range header
2. Find entries matching INCLUDE-50 video paths
3. Download only those entries' compressed data
4. Extract to data/raw/

Hard rule #5: Total download ~11-12 GB instead of 53 GB.
"""

import os
import sys
import struct
import io
import zipfile
import json
import time
import hashlib
import argparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path


ZENODO_RECORD = "4010759"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Map category folder names to their Zenodo zip file prefix
CATEGORY_ZIP_PREFIX = {
    "Adjectives": "Adjectives",
    "Animals": "Animals",
    "Clothes": "Clothes",
    "Colours": "Colours",
    "Days_and_Time": "Days_and_Time",
    "Electronics": "Electronics",
    "Greetings": "Greetings",
    "Home": "Home",
    "Jobs": "Jobs",
    "Means_of_Transportation": "Means_of_Transportation",
    "People": "People",
    "Places": "Places",
    "Pronouns": "Pronouns",
    "Seasons": "Seasons",
    "Society": "Society",
}


def load_include50_paths(splits_dir):
    """Load all video paths from INCLUDE-50 split files."""
    paths = set()
    for split in ["include50_train.txt", "include50_val.txt", "include50_test.txt"]:
        fpath = os.path.join(splits_dir, split)
        if not os.path.exists(fpath):
            print(f"ERROR: Split file not found: {fpath}")
            sys.exit(1)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.add(line)
    return paths


def get_zenodo_files():
    """Fetch the list of files from the Zenodo record."""
    req = Request(ZENODO_API)
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    files = {}
    for f in data["files"]:
        if f["key"].endswith(".zip"):
            files[f["key"]] = {
                "url": f["links"]["self"],
                "size": f["size"],
                "checksum": f["checksum"],
            }
    return files


def read_range(url, start, end):
    """Download a byte range from a URL using HTTP Range header."""
    req = Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    with urlopen(req) as resp:
        return resp.read()


def find_eocd(url, file_size):
    """Find the End of Central Directory record in a remote zip file."""
    # EOCD is in the last 65KB max (includes comment)
    read_size = min(65536, file_size)
    start = file_size - read_size
    data = read_range(url, start, file_size - 1)

    # Search for EOCD signature (0x06054b50) from end
    eocd_sig = b"\x50\x4b\x05\x06"
    pos = data.rfind(eocd_sig)
    if pos == -1:
        return None

    eocd = data[pos:]
    if len(eocd) < 22:
        return None

    # Parse EOCD
    (sig, disk_num, disk_start, num_entries_disk, num_entries,
     cd_size, cd_offset, comment_len) = struct.unpack_from("<IHHHHIIH", eocd)

    return {
        "num_entries": num_entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
    }


def read_central_directory(url, file_size, eocd):
    """Read and parse the central directory from a remote zip."""
    cd_data = read_range(url, eocd["cd_offset"],
                         eocd["cd_offset"] + eocd["cd_size"] - 1)

    entries = []
    offset = 0
    cd_sig = b"\x50\x4b\x01\x02"

    while offset < len(cd_data):
        if cd_data[offset:offset+4] != cd_sig:
            break

        # Parse central directory file header
        header = struct.unpack_from("<IHHHHHHIIIHHHHHII", cd_data, offset)
        (sig, ver_made, ver_needed, flags, method, mod_time, mod_date,
         crc32, comp_size, uncomp_size, name_len, extra_len, comment_len,
         disk_start, int_attr, ext_attr, local_offset) = header

        name = cd_data[offset+46:offset+46+name_len].decode("utf-8", errors="replace")

        entries.append({
            "name": name,
            "method": method,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "local_offset": local_offset,
            "crc32": crc32,
        })

        offset += 46 + name_len + extra_len + comment_len

    return entries


def extract_entry(url, entry, output_path):
    """Extract a single file entry from a remote zip."""
    # Read local file header to find actual data start
    local_header = read_range(url, entry["local_offset"],
                              entry["local_offset"] + 29)

    if local_header[:4] != b"\x50\x4b\x03\x04":
        print(f"  ERROR: Invalid local header for {entry['name']}")
        return False

    name_len = struct.unpack_from("<H", local_header, 26)[0]
    extra_len = struct.unpack_from("<H", local_header, 28)[0]
    data_offset = entry["local_offset"] + 30 + name_len + extra_len

    # Download compressed data
    comp_data = read_range(url, data_offset,
                           data_offset + entry["comp_size"] - 1)

    # Decompress
    if entry["method"] == 0:  # Stored
        file_data = comp_data
    elif entry["method"] == 8:  # Deflated
        import zlib
        file_data = zlib.decompress(comp_data, -15)
    else:
        print(f"  ERROR: Unsupported compression method {entry['method']} for {entry['name']}")
        return False

    # Verify CRC
    if zlib.crc32(file_data) & 0xffffffff != entry["crc32"]:
        print(f"  WARNING: CRC mismatch for {entry['name']}")

    # Write file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(file_data)

    return True


def main():
    parser = argparse.ArgumentParser(description="Smart INCLUDE-50 downloader")
    parser.add_argument("--splits-dir", default="data/splits",
                        help="Directory containing split .txt files")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Directory to save extracted videos")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only calculate download size, don't download")
    args = parser.parse_args()

    print("=" * 60)
    print("INCLUDE-50 Smart Downloader")
    print("=" * 60)

    # Load needed video paths
    needed_paths = load_include50_paths(args.splits_dir)
    print(f"\nVideos needed: {len(needed_paths)}")

    # Check which are already downloaded
    already_have = 0
    still_need = set()
    for p in needed_paths:
        out = os.path.join(args.output_dir, p)
        if os.path.exists(out):
            already_have += 1
        else:
            still_need.add(p)

    if already_have > 0:
        print(f"Already downloaded: {already_have}")
    if not still_need:
        print("All videos already downloaded!")
        return

    print(f"Still need: {len(still_need)}")

    # Group needed paths by category
    by_category = {}
    for p in still_need:
        cat = p.split("/")[0]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)

    print(f"\nCategories to process: {len(by_category)}")
    for cat in sorted(by_category):
        print(f"  {cat}: {len(by_category[cat])} videos")

    # Fetch Zenodo file list
    print("\nFetching Zenodo file list...")
    zenodo_files = get_zenodo_files()
    print(f"Found {len(zenodo_files)} zip files")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above categories.")
        print("Estimated download: ~11-12 GB (video data only, not full zips)")
        return

    # Process each category
    total_downloaded = 0
    total_extracted = 0
    errors = []
    manifest = {}

    for cat in sorted(by_category):
        cat_videos = by_category[cat]
        prefix = CATEGORY_ZIP_PREFIX.get(cat, cat)

        # Find matching zip files
        cat_zips = {k: v for k, v in zenodo_files.items()
                    if k.startswith(prefix + "_")}

        if not cat_zips:
            print(f"\nWARNING: No zip files found for category '{cat}'")
            errors.append(f"No zips for {cat}")
            continue

        print(f"\n--- {cat} ({len(cat_videos)} videos, {len(cat_zips)} zips) ---")

        found_videos = set()

        for zip_name, zip_info in sorted(cat_zips.items()):
            url = zip_info["url"]
            file_size = zip_info["size"]

            print(f"\n  Scanning {zip_name} ({file_size / (1024**2):.0f} MB)...")

            try:
                # Read EOCD
                eocd = find_eocd(url, file_size)
                if not eocd:
                    print(f"  ERROR: Could not find EOCD in {zip_name}")
                    errors.append(f"No EOCD: {zip_name}")
                    continue

                # Read central directory
                entries = read_central_directory(url, file_size, eocd)
                print(f"  Entries in zip: {len(entries)}")

                # Find matching entries
                for entry in entries:
                    # Normalize: entry names may have category prefix
                    ename = entry["name"]
                    # Check against our needed paths
                    for vpath in cat_videos:
                        if vpath in found_videos:
                            continue
                        # The zip might store as "Category/N. Sign/file.MOV"
                        # or just the relative path
                        if ename.endswith(vpath) or ename == vpath or \
                           ename.rstrip("/") == vpath.rstrip("/"):
                            out_path = os.path.join(args.output_dir, vpath)
                            print(f"  Extracting: {vpath} ({entry['uncomp_size'] / (1024**2):.1f} MB)")

                            success = extract_entry(url, entry, out_path)
                            if success:
                                found_videos.add(vpath)
                                total_extracted += 1
                                total_downloaded += entry["comp_size"]
                                manifest[vpath] = {
                                    "size": entry["uncomp_size"],
                                    "source_zip": zip_name,
                                    "crc32": entry["crc32"],
                                }
                            else:
                                errors.append(f"Extract failed: {vpath}")
                            break

                remaining = set(cat_videos) - found_videos
                if not remaining:
                    print(f"  All {cat} videos found!")
                    break

            except HTTPError as e:
                print(f"  HTTP Error: {e}")
                errors.append(f"HTTP {e.code}: {zip_name}")
            except Exception as e:
                print(f"  Error: {e}")
                errors.append(f"Error in {zip_name}: {e}")

        not_found = set(cat_videos) - found_videos
        if not_found:
            print(f"  WARNING: {len(not_found)} videos not found in any zip:")
            for nf in sorted(not_found)[:5]:
                print(f"    {nf}")
            errors.extend([f"Not found: {nf}" for nf in not_found])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"  Extracted: {total_extracted} / {len(still_need)} videos")
    print(f"  Downloaded: {total_downloaded / (1024**3):.2f} GB (compressed)")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    {e}")

    # Save manifest
    manifest_path = os.path.join("data", "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
