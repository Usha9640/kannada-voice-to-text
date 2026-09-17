import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pydub import AudioSegment
from pydub.generators import Sine
from transcribe import (
    validate_file,
    split_audio,
    transcribe_chunk,
    transcribe_audio,
    MAX_FILE_SIZE_MB,
    MAX_CHUNK_MS,
    MIN_CHUNK_MS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_audio(duration_ms: int, freq: int = 440) -> AudioSegment:
    return Sine(freq).to_audio_segment(duration=duration_ms).set_frame_rate(16000).set_channels(1)


def audio_to_wav_file(audio: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        return f.name


# ── validate_file ─────────────────────────────────────────────────────────────

def test_validate_file_ok():
    small_bytes = b"x" * (10 * 1024 * 1024)  # 10 MB
    assert validate_file(small_bytes, "test.wav") is None


def test_validate_file_too_large():
    big_bytes = b"x" * ((MAX_FILE_SIZE_MB + 1) * 1024 * 1024)
    result = validate_file(big_bytes, "big.wav")
    assert result is not None
    assert "MB" in result


# ── split_audio ───────────────────────────────────────────────────────────────

def test_split_audio_short_clip():
    audio = make_audio(5000)
    chunks = split_audio(audio)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) >= MIN_CHUNK_MS
        assert len(chunk) <= MAX_CHUNK_MS


def test_split_audio_long_clip():
    audio = make_audio(90000)  # 90s — must be hard-split
    chunks = split_audio(audio)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= MAX_CHUNK_MS


def test_split_audio_filters_short_silence():
    silence = AudioSegment.silent(duration=200).set_frame_rate(16000).set_channels(1)
    chunks = split_audio(silence)
    assert all(len(c) >= MIN_CHUNK_MS for c in chunks)


# ── transcribe_chunk ──────────────────────────────────────────────────────────

def test_transcribe_chunk_success():
    audio = make_audio(3000)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        chunk_path = f.name

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"transcript": "ನಮಸ್ಕಾರ"}

    try:
        with patch("transcribe.requests.post", return_value=mock_resp):
            result, warn = transcribe_chunk(audio, chunk_path, 0, "fake_key")
        assert result == "ನಮಸ್ಕಾರ"
        assert warn is None
    finally:
        os.remove(chunk_path)


def test_transcribe_chunk_api_error():
    audio = make_audio(3000)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        chunk_path = f.name

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    try:
        with patch("transcribe.requests.post", return_value=mock_resp):
            result, warn = transcribe_chunk(audio, chunk_path, 0, "fake_key")
        assert result is None
        assert warn is not None
        assert "skipped" in warn
    finally:
        os.remove(chunk_path)


def test_transcribe_chunk_timeout():
    import requests as req
    audio = make_audio(3000)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        chunk_path = f.name

    try:
        with patch("transcribe.requests.post", side_effect=req.exceptions.Timeout):
            result, warn = transcribe_chunk(audio, chunk_path, 0, "fake_key")
        assert result is None
        assert warn is not None
    finally:
        os.remove(chunk_path)


# ── transcribe_audio ──────────────────────────────────────────────────────────

def test_transcribe_audio_full_pipeline():
    audio = make_audio(10000)
    wav_path = audio_to_wav_file(audio)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"transcript": "ಹಲೋ ವರ್ಲ್ಡ್"}

    progress_calls = []

    def fake_progress(current, total, msg):
        progress_calls.append((current, total))

    try:
        with patch("transcribe.requests.post", return_value=mock_resp):
            transcript, warnings = transcribe_audio(wav_path, "fake_key", fake_progress)
        assert "ಹಲೋ ವರ್ಲ್ಡ್" in transcript
        assert warnings == []
        assert len(progress_calls) > 0
    finally:
        os.remove(wav_path)


def test_transcribe_audio_skips_failed_chunks():
    audio = make_audio(10000)
    wav_path = audio_to_wav_file(audio)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "error"

    try:
        with patch("transcribe.requests.post", return_value=mock_resp):
            transcript, warnings = transcribe_audio(wav_path, "fake_key")
        assert transcript == ""
        assert len(warnings) > 0
    finally:
        os.remove(wav_path)
