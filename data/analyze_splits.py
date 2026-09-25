"""Analyze INCLUDE-50 split files to determine download requirements."""
import json
import os

# Read split files
splits = {}
for name in ["include50_train", "include50_test", "include50_val"]:
    path = os.path.join("data", "splits", f"{name}.txt")
    with open(path) as f:
        splits[name] = [l.strip() for l in f if l.strip()]

print(f"Train: {len(splits['include50_train'])} videos")
print(f"Val:   {len(splits['include50_val'])} videos")
print(f"Test:  {len(splits['include50_test'])} videos")

# Get unique categories and signs
all_paths = splits["include50_train"] + splits["include50_val"] + splits["include50_test"]
categories = set()
signs = set()
for p in all_paths:
    parts = p.split("/")
    categories.add(parts[0])
    if len(parts) > 1:
        signs.add(parts[1])

print(f"\nTotal videos: {len(all_paths)}")
print(f"Categories needed: {len(categories)}")
for c in sorted(categories):
    count = sum(1 for p in all_paths if p.startswith(c + "/"))
    print(f"  {c}: {count} videos")
print(f"\nUnique signs: {len(signs)}")

# Read label map
lm_path = os.path.join("data", "label_maps", "label_map_include50.json")
with open(lm_path) as f:
    lm = json.load(f)
print(f"\nLabel map: {len(lm)} classes")
for k, v in sorted(lm.items(), key=lambda x: int(x[1])):
    print(f"  {v}: {k}")

# Zenodo zip files and sizes (from API)
zenodo_files = {
    "Adjectives_1of8.zip": 1303983457, "Adjectives_2of8.zip": 1407281940,
    "Adjectives_3of8.zip": 1425808613, "Adjectives_4of8.zip": 1214968173,
    "Adjectives_5of8.zip": 1310663669, "Adjectives_6of8.zip": 1247813680,
    "Adjectives_7of8.zip": 1253415149, "Adjectives_8of8.zip": 834989744,
    "Animals_1of2.zip": 1819837734, "Animals_2of2.zip": 1069027692,
    "Clothes_1of2.zip": 1387609869, "Clothes_2of2.zip": 1396763182,
    "Colours_1of2.zip": 1268663334, "Colours_2of2.zip": 1452446819,
    "Days_and_Time_1of3.zip": 1244115203, "Days_and_Time_2of3.zip": 1016797379,
    "Days_and_Time_3of3.zip": 852955661,
    "Electronics_1of2.zip": 926290038, "Electronics_2of2.zip": 824110880,
    "Greetings_1of2.zip": 1573733886, "Greetings_2of2.zip": 1208700458,
    "Home_1of4.zip": 1248459133, "Home_2of4.zip": 1338509691,
    "Home_3of4.zip": 1080038303, "Home_4of4.zip": 873459938,
    "Jobs_1of2.zip": 1503117419, "Jobs_2of2.zip": 1650647791,
    "Means_of_Transportation_1of2.zip": 1846003966, "Means_of_Transportation_2of2.zip": 1606937936,
    "People_1of5.zip": 1327653833, "People_2of5.zip": 1251273463,
    "People_3of5.zip": 1586966004, "People_4of5.zip": 1303778856,
    "People_5of5.zip": 1301122063,
    "Places_1of4.zip": 1394196824, "Places_2of4.zip": 1355857452,
    "Places_3of4.zip": 1466179266, "Places_4of4.zip": 1058186818,
    "Pronouns_1of2.zip": 1419853451, "Pronouns_2of2.zip": 946837765,
    "Seasons_1of1.zip": 1251955665,
    "Society_1of3.zip": 1436640862, "Society_2of3.zip": 1358587307,
    "Society_3of3.zip": 1105579609,
}

# Which zips match needed categories
needed_zips = {}
for zname, zsize in zenodo_files.items():
    cat = zname.rsplit("_", 1)[0]
    # Handle multi-word categories
    if cat.replace("_", " ") in categories or cat in categories:
        needed_zips[zname] = zsize
    else:
        # Try matching with underscores replaced
        for c in categories:
            if cat == c.replace(" ", "_") or cat.startswith(c.replace(" ", "_")):
                needed_zips[zname] = zsize
                break

total_all = sum(zenodo_files.values()) / (1024**3)
total_needed = sum(needed_zips.values()) / (1024**3)

print(f"\n--- Download Analysis ---")
print(f"Total Zenodo dataset: {total_all:.1f} GB ({len(zenodo_files)} zips)")
print(f"Zips matching INCLUDE-50 categories: {total_needed:.1f} GB ({len(needed_zips)} zips)")
print(f"\nNeeded zips:")
for z in sorted(needed_zips):
    print(f"  {z}: {needed_zips[z] / (1024**3):.2f} GB")
