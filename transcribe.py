import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pydub import AudioSegment
from pydub.silence import split_on_silence

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"

MAX_CHUNK_MS = 28000
MIN_CHUNK_MS = 500
SILENCE_THRESH_DB = -40
MIN_SILENCE_MS = 500
MAX_FILE_SIZE_MB = 50


def validate_file(file_bytes: bytes, filename: str) -> str | None:
    """Return error message string if invalid, else None."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"File is {size_mb:.1f} MB — maximum allowed is {MAX_FILE_SIZE_MB} MB."
    return None


def convert_to_wav(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    out_path = input_path + "_converted.wav"
    audio.export(out_path, format="wav")
    return out_path, len(audio)  # return duration in ms too


def split_audio(audio: AudioSegment) -> list[AudioSegment]:
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=200,
    )
    if not chunks:
        chunks = [audio]

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHUNK_MS:
            final_chunks.append(chunk)
        else:
            for start in range(0, len(chunk), MAX_CHUNK_MS):
                final_chunks.append(chunk[start:start + MAX_CHUNK_MS])

    return [c for c in final_chunks if len(c) >= MIN_CHUNK_MS]


def transcribe_chunk(chunk: AudioSegment, chunk_path: str, index: int, api_key: str) -> tuple[str | None, str | None]:
    """Returns (transcript, warning_message). warning_message is None on success."""
    chunk.export(chunk_path, format="wav")

    for attempt in range(2):
        try:
            with open(chunk_path, "rb") as f:
                response = requests.post(
                    SARVAM_API_URL,
                    headers={"api-subscription-key": api_key},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={
                        "language_code": "kn-IN",
                        "model": "saarika:v2.5",
                        "with_disfluencies": "false",
                    },
                    timeout=30,
                )
            if response.status_code == 200:
                return response.json().get("transcript", "").strip(), None
            elif response.status_code == 429:
                time.sleep(2)
                continue
            else:
                return None, f"Chunk {index + 1}: API error {response.status_code} — skipped."
        except requests.exceptions.Timeout:
            if attempt == 0:
                continue
            return None, f"Chunk {index + 1}: timed out — skipped."
        except Exception as e:
            return None, f"Chunk {index + 1}: {e} — skipped."
    return None, f"Chunk {index + 1}: failed after retries — skipped."


MAX_WORKERS = 4  # parallel API calls — safe limit for Sarvam rate limits


def _process_chunk(args):
    """Worker function for parallel execution."""
    i, chunk, api_key = args
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        chunk_path = tmp.name
    try:
        text, warn = transcribe_chunk(chunk, chunk_path, i, api_key)
        return i, text, warn
    finally:
        try:
            os.remove(chunk_path)
        except OSError:
            pass


def transcribe_audio(wav_path: str, api_key: str, progress_callback=None) -> tuple[str, list[str]]:
    """
    Returns (transcript, warnings).
    Chunks are processed in parallel (up to MAX_WORKERS at a time).
    Results are re-ordered by original chunk index to preserve transcript order.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunks = split_audio(audio)
    total = len(chunks)
    results = {}   # {index: text}
    warnings = []
    completed = 0

    if progress_callback:
        progress_callback(0, total, f"Transcribing {total} chunk(s) in parallel...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_chunk, (i, chunk, api_key)): i
                   for i, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            i, text, warn = future.result()
            if text:
                results[i] = text
            if warn:
                warnings.append(warn)
            completed += 1
            if progress_callback:
                progress_callback(completed, total, f"Completed {completed} of {total} chunks...")

    if progress_callback:
        progress_callback(total, total, "Done!")

    # Re-join in original order
    return " ".join(results[i] for i in sorted(results)), warnings
