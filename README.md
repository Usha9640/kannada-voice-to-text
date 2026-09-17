# Kannada Voice-to-Text Converter

A Streamlit web app that transcribes speech into Kannada (ಕನ್ನಡ) text using OpenAI's Whisper model.

## 1. Prerequisites
- Python 3.9+
- ffmpeg installed on your system (required for audio conversion)
  - Windows: `choco install ffmpeg` (or download from ffmpeg.org and add to PATH)
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## 2. Setup

```bash
# 1. Extract this zip and move into the folder
cd kannada_app

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

## 3. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

On first run, it will download the Whisper model (~1-2 GB for `whisper-small`). This can take a few minutes — after that, it's cached and loads instantly on future runs.

## 4. Using the app
- **Upload tab**: upload an MP3/WAV/M4A file of someone speaking Kannada, click "Transcribe uploaded file"
- **Record tab**: click the mic button, speak in Kannada, stop recording, click "Transcribe recorded audio"
- The transcribed Kannada text appears in a copyable text box, with a download button for a `.txt` file

## 5. Tips
- For faster transcription on a CPU-only machine, open `app.py` and change:
  ```python
  MODEL_NAME = "openai/whisper-small"
  ```
  to:
  ```python
  MODEL_NAME = "openai/whisper-base"
  ```
- A GPU (CUDA) will be used automatically if available — much faster than CPU.
- For best accuracy, use clear audio with minimal background noise.

## 6. Troubleshooting
- **"ffmpeg not found" error**: make sure ffmpeg is installed and on your system PATH. Test with `ffmpeg -version` in your terminal.
- **Model download is slow/fails**: check your internet connection; the model downloads from Hugging Face on first run only.
- **st.audio_input not found**: upgrade Streamlit with `pip install -U streamlit` (needs version 1.31+).
