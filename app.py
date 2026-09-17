import streamlit as st
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from transcribe import convert_to_wav, transcribe_audio, validate_file, MAX_FILE_SIZE_MB

load_dotenv()

st.set_page_config(page_title="Kannada Voice-to-Text Converter", page_icon="🎙️", layout="centered")

# Read API key from Streamlit Cloud secrets or local .env
# Safely read from Streamlit Cloud secrets or local .env
try:
    API_KEY = st.secrets.get("SARVAM_API_KEY") or os.getenv("SARVAM_API_KEY")
except Exception:
    API_KEY = os.getenv("SARVAM_API_KEY")

# Session state for transcription history
if "history" not in st.session_state:
    st.session_state.history = []  # list of {time, filename, transcript}


def add_to_history(filename: str, transcript: str):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "filename": filename,
        "transcript": transcript,
    })
    st.session_state.history = st.session_state.history[:5]  # keep last 5


def show_result(transcript: str, filename: str):
    st.success("✅ Transcription complete!")
    add_to_history(filename, transcript)
    st.subheader("📝 Kannada Transcription")
    st.text_area("Copy your transcribed text below:", value=transcript, height=200, key=f"result_{filename}")
    st.download_button(
        "⬇️ Download as .txt",
        data=transcript,
        file_name="kannada_transcription.txt",
        mime="text/plain",
    )


def process_audio(audio_file_path: str, filename: str):
    converted_path = None
    try:
        with st.spinner("🔄 Converting audio..."):
            converted_path, duration_ms = convert_to_wav(audio_file_path)

        duration_sec = duration_ms / 1000
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        chunks_estimate = max(1, int(duration_sec // 28) + 1)

        st.info(
            f"⏱️ Audio duration: **{minutes}m {seconds}s** — "
            f"estimated **{chunks_estimate} chunk(s)** to transcribe."
        )

        progress_bar = st.progress(0, text="Starting...")

        def progress_callback(current, total, message):
            progress_bar.progress(current / total if total > 0 else 1.0, text=message)

        transcript, warnings = transcribe_audio(converted_path, API_KEY, progress_callback)

        for warn in warnings:
            st.warning(f"⚠️ {warn}")

        if transcript.strip():
            show_result(transcript, filename)
        else:
            st.warning("⚠️ No speech detected. Please try again with a clearer recording.")

    except FileNotFoundError:
        st.error("❌ Audio conversion failed. Make sure **ffmpeg** is installed.")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
    finally:
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except OSError:
                pass


def main():
    st.title("🎙️ Kannada Voice-to-Text Converter")
    st.markdown(
        "Transcribe **Kannada (ಕನ್ನಡ)** speech — supports short clips and long recordings (5+ minutes). "
        "Also handles Kannada–English mixed speech."
    )

    if not API_KEY:
        st.error("❌ SARVAM_API_KEY not found. Add it to `.env` locally or Streamlit Cloud secrets.")
        st.stop()

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎤 Record Audio", "🕓 History"])

    # ---------------- TAB 1: FILE UPLOAD ----------------
    with tab1:
        st.caption(f"Supported: MP3, WAV, M4A, OGG, FLAC — max {MAX_FILE_SIZE_MB} MB")
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            error = validate_file(file_bytes, uploaded_file.name)
            if error:
                st.error(f"❌ {error}")
            else:
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file_bytes)
                    audio_file_path = tmp.name

                st.audio(file_bytes)
                if st.button("🔁 Transcribe Uploaded File", type="primary", key="upload_btn"):
                    process_audio(audio_file_path, uploaded_file.name)
                    try:
                        os.remove(audio_file_path)
                    except OSError:
                        pass

    # ---------------- TAB 2: RECORD AUDIO ----------------
    with tab2:
        st.write("Click the mic to record, stop when done, then transcribe.")
        recorded_audio = None
        try:
            recorded_audio = st.audio_input("Record your voice", label_visibility="collapsed")
        except AttributeError:
            st.warning("⚠️ Please upgrade Streamlit: `pip install -U streamlit`")

        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_file_path = tmp.name

            st.audio(audio_bytes)
            if st.button("🔁 Transcribe Recorded Audio", type="primary", key="record_btn"):
                process_audio(audio_file_path, "recorded_audio.wav")
                try:
                    os.remove(audio_file_path)
                except OSError:
                    pass

    # ---------------- TAB 3: HISTORY ----------------
    with tab3:
        if not st.session_state.history:
            st.info("No transcriptions yet. Transcribe something to see history here.")
        else:
            for i, entry in enumerate(st.session_state.history):
                with st.expander(f"🕓 {entry['time']} — {entry['filename']}"):
                    st.text_area("Transcript", value=entry["transcript"], height=150, key=f"hist_{i}")
                    st.download_button(
                        "⬇️ Download",
                        data=entry["transcript"],
                        file_name=f"transcript_{entry['time'].replace(':', '')}.txt",
                        mime="text/plain",
                        key=f"dl_{i}",
                    )
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()


if __name__ == "__main__":
    main()
