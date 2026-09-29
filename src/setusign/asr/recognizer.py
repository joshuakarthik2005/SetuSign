"""
Speech-to-Text (ASR) Module for SetuSign
=========================================
Provides real-time speech recognition using faster-whisper,
producing large on-screen captions for deaf users at service counters.

Runs entirely on-device using CTranslate2 (CPU/GPU, no cloud).
Supports Hindi and English speech.
"""

import os
import sys
import time
import queue
import threading
import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class SpeechRecognizer:
    """
    Real-time speech recognizer using faster-whisper.
    
    Runs in a background thread, feeding recognized text
    to a queue for the main application to consume.
    """

    def __init__(self, model_size="tiny", language="en", device="cpu",
                 compute_type="int8"):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium)
            language: Language code (en, hi, etc.)
            device: cpu or cuda
            compute_type: int8, float16, float32
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type

        self.model = None
        self.is_loaded = False
        self.caption_queue = queue.Queue(maxsize=100)
        self.current_caption = ""
        self.caption_history = []

        if HAS_WHISPER:
            self._load_model()
        else:
            print("WARNING: faster-whisper not installed. ASR disabled.")

    def _load_model(self):
        """Load the Whisper model."""
        print(f"Loading Whisper {self.model_size} model...")
        t0 = time.time()
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self.is_loaded = True
            elapsed = time.time() - t0
            print(f"  Whisper loaded in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Failed to load Whisper: {e}")
            self.is_loaded = False

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        Transcribe an audio chunk.
        
        Args:
            audio_data: numpy array of audio samples (float32, mono)
            sample_rate: audio sample rate (default 16000)
        
        Returns:
            Transcribed text string
        """
        if not self.is_loaded:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=1,  # Fast decoding
                vad_filter=True,  # Filter out silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            text = " ".join(text_parts)
            if text:
                self.current_caption = text
                self.caption_history.append({
                    "text": text,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "language": info.language if info else self.language,
                })
                self.caption_queue.put(text)

            return text

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def get_latest_caption(self):
        """Get the most recent caption (non-blocking)."""
        try:
            text = self.caption_queue.get_nowait()
            self.current_caption = text
            return text
        except queue.Empty:
            return self.current_caption

    def get_model_info(self):
        """Return model information."""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "is_loaded": self.is_loaded,
            "has_whisper": HAS_WHISPER,
        }


class MicrophoneCapture:
    """
    Captures audio from microphone in a background thread.
    Uses sounddevice or falls back to pyaudio.
    """

    def __init__(self, sample_rate=16000, chunk_duration=3.0):
        """
        Args:
            sample_rate: Audio sample rate
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_running = False
        self._thread = None

    def start(self):
        """Start capturing audio."""
        try:
            import sounddevice as sd
            self._backend = "sounddevice"
        except ImportError:
            print("WARNING: sounddevice not installed. Microphone capture disabled.")
            print("  Install with: pip install sounddevice")
            self._backend = None
            return False

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        """Background thread for audio capture."""
        import sounddevice as sd

        while self.is_running:
            try:
                audio = sd.rec(
                    self.chunk_size,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                self.audio_queue.put(audio.flatten())
            except Exception as e:
                print(f"Audio capture error: {e}")
                time.sleep(0.5)

    def get_audio_chunk(self):
        """Get the next audio chunk (blocking)."""
        try:
            return self.audio_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        """Stop capturing audio."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)


class CaptionService:
    """
    High-level service combining microphone capture and speech recognition.
    Produces continuous captions for display.
    """

    def __init__(self, model_size="tiny", language="en"):
        self.recognizer = SpeechRecognizer(model_size=model_size, language=language)
        self.mic = MicrophoneCapture()
        self.is_running = False
        self._thread = None
        self.current_caption = ""
        self.caption_history = []

    def start(self):
        """Start the caption service."""
        if not self.recognizer.is_loaded:
            print("Cannot start captions: Whisper not loaded")
            return False

        if not self.mic.start():
            print("Cannot start captions: Microphone not available")
            return False

        self.is_running = True
        self._thread = threading.Thread(target=self._caption_loop, daemon=True)
        self._thread.start()
        print("Caption service started")
        return True

    def _caption_loop(self):
        """Background loop: capture audio -> transcribe -> update captions."""
        while self.is_running:
            audio = self.mic.get_audio_chunk()
            if audio is not None:
                text = self.recognizer.transcribe_audio(audio)
                if text:
                    self.current_caption = text
                    self.caption_history.append(text)
                    # Keep last 10 captions
                    if len(self.caption_history) > 10:
                        self.caption_history.pop(0)

    def get_caption(self):
        """Get current caption text."""
        return self.current_caption

    def get_history(self):
        """Get caption history."""
        return self.caption_history

    def stop(self):
        """Stop the caption service."""
        self.is_running = False
        self.mic.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
