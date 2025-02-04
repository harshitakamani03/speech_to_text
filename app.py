import streamlit as st
from streamlit_audiorec import st_audiorec
import requests
import openai
import psycopg2
import psycopg2.extras
import os
import time
import concurrent.futures
import uuid
import boto3
import io

# ----------------------------------------------------------------------------
# 1) READ CREDENTIALS & CONFIG FROM STREAMLIT SECRETS
# ----------------------------------------------------------------------------
DEEPGRAM_API_KEY     = st.secrets["DEEPGRAM_API_KEY"]
OPENAI_API_KEY       = st.secrets["OPENAI_API_KEY"]
ASSEMBLYAI_API_KEY   = st.secrets["ASSEMBLYAI_API_KEY"]

DATABASE_URL         = st.secrets["DATABASE_URL"]

AWS_ACCESS_KEY_ID     = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION    = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")  # optional
AWS_S3_BUCKET_NAME    = st.secrets["AWS_S3_BUCKET_NAME"]

# ----------------------------------------------------------------------------
# 2) SET DEPARTMENT LIST
# ----------------------------------------------------------------------------
DEPARTMENTS = [
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
    "Neurology",
    "Rheumatology",
    "Dermatology",
    "Nephrology",
    "Hematology",
    "Infectious Diseases",
    "Psychiatry",
    "Pediatrics",
    "Orthopedics",
    "Ophthalmology",
    "Otolaryngology",
    "Gynecology",
    "Urology",
    "Oncology",
    "General Medicine",
    "Endocrinology",
]

# ----------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def upload_audio_to_s3(audio_bytes: bytes) -> str:
    """Uploads audio bytes to AWS S3, returns the public or presigned URL."""
    if not AWS_S3_BUCKET_NAME:
        raise ValueError("Missing AWS_S3_BUCKET_NAME in secrets.")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

    unique_id = str(uuid.uuid4())
    object_key = f"audio_uploads/{unique_id}.wav"

    audio_stream = io.BytesIO(audio_bytes)
    s3.upload_fileobj(audio_stream, AWS_S3_BUCKET_NAME, object_key)

    # Construct a direct URL (public if your bucket is public, or presigned otherwise)
    audio_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{object_key}"
    return audio_url


def transcribe_deepgram(audio_bytes: bytes) -> str:
    """Transcribe WAV bytes with Deepgram's medical model."""
    if not DEEPGRAM_API_KEY:
        return "Missing Deepgram API key."

    url = "https://api.deepgram.com/v1/listen?model=nova-2-medical"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    try:
        response = requests.post(url, headers=headers, data=audio_bytes)
        response.raise_for_status()
        result = response.json()
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript.strip()
    except Exception as e:
        st.error(f"Deepgram transcription error: {e}")
        return ""


def transcribe_whisper(audio_bytes: bytes) -> str:
    """Transcribe WAV bytes with OpenAI's Whisper API."""
    if not OPENAI_API_KEY:
        return "Missing OpenAI API key."

    openai.api_key = OPENAI_API_KEY
    try:
        # Use in-memory buffer (OpenAI needs a file-like object with a name attribute).
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "temp_audio.wav"
        transcript_data = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript_data["text"].strip()
    except Exception as e:
        st.error(f"Whisper transcription error: {e}")
        return ""


def transcribe_assemblyai(audio_bytes: bytes) -> str:
    """Transcribe WAV bytes with AssemblyAI (batch async)."""
    if not ASSEMBLYAI_API_KEY:
        return "Missing AssemblyAI API key."

    temp_file = "temp_assembly.wav"
    try:
        # Write temporary file
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        # Upload
        with open(temp_file, "rb") as f_data:
            data = f_data.read()
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        upload_resp = requests.post(upload_url, headers=headers, data=data)
        upload_resp.raise_for_status()
        upload_result = upload_resp.json()

        if "upload_url" not in upload_result:
            return "AssemblyAI upload response invalid."

        audio_url = upload_result["upload_url"]

        # Start transcription
        transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
        json_payload = {"audio_url": audio_url}
        start_resp = requests.post(
            transcript_endpoint,
            headers={"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"},
            json=json_payload
        )
        start_resp.raise_for_status()
        transcript_id = start_resp.json()["id"]

        # Poll for completion
        polling_endpoint = f"{transcript_endpoint}/{transcript_id}"
        while True:
            poll_resp = requests.get(polling_endpoint, headers={"authorization": ASSEMBLYAI_API_KEY})
            poll_resp.raise_for_status()
            status_data = poll_resp.json()
            status = status_data["status"]

            if status == "completed":
                return status_data["text"].strip()
            elif status == "error":
                st.error("AssemblyAI transcription error: " + status_data.get("error", "Unknown"))
                return ""
            else:
                time.sleep(1)

    except Exception as e:
        st.error(f"AssemblyAI error: {e}")
        return ""


def transcribe_all_in_parallel(audio_bytes: bytes):
    """
    Runs the three transcription functions (Deepgram, Whisper, AssemblyAI)
    in parallel threads for faster processing.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_deepgram   = executor.submit(transcribe_deepgram, audio_bytes)
        future_whisper    = executor.submit(transcribe_whisper, audio_bytes)
        future_assemblyai = executor.submit(transcribe_assemblyai, audio_bytes)

        deepgram_text   = future_deepgram.result()
        whisper_text    = future_whisper.result()
        assemblyai_text = future_assemblyai.result()

    return deepgram_text, whisper_text, assemblyai_text


def save_transcripts_to_postgres(
    dept: str,
    audio_url: str,
    deepgram_text: str,
    whisper_text: str,
    assemblyai_text: str,
    chosen_engine: str
):
    """Inserts a record into the 'transcripts' table in PostgreSQL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        insert_query = """
            INSERT INTO transcripts (
                dept,
                audio_url,
                deepgram_transcript,
                whisper_transcript,
                assemblyai_transcript,
                chosen_transcript
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        cur.execute(
            insert_query,
            (
                dept,
                audio_url,
                deepgram_text,
                whisper_text,
                assemblyai_text,
                chosen_engine
            )
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error saving transcript to PostgreSQL: {e}")


# ----------------------------------------------------------------------------
# 4) MAIN STREAMLIT APP
# ----------------------------------------------------------------------------
def main():
    # --- Custom CSS ---
    st.markdown("""
        <style>
        body {
            background: #e6edf2;
        }
        .main .block-container {
            background-color: #fff;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            max-width: 800px;
            margin: 2rem auto;
        }
        h1 {
            margin-bottom: 1.2rem !important;
        }
        .dept-label-custom {
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
        }
        .instruction-box {
            background-color: #f9f9f9;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #2b78e4;
            border-radius: 4px;
            margin-top: 0.3rem;
        }
        .stRadio label {
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- App Title ---
    st.title("Voice to Text Testing")

    # --- Check for missing credentials ---
    missing_keys = []
    if not DEEPGRAM_API_KEY:
        missing_keys.append("Deepgram")
    if not OPENAI_API_KEY:
        missing_keys.append("OpenAI")
    if not ASSEMBLYAI_API_KEY:
        missing_keys.append("AssemblyAI")

    if missing_keys:
        st.error(f"Missing API keys for: {', '.join(missing_keys)}. Please add them in Streamlit Secrets.")
        return

    if not DATABASE_URL:
        st.error("Missing DATABASE_URL in secrets.")
        return

    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_S3_BUCKET_NAME):
        st.error("Missing AWS credentials or bucket name in secrets.")
        return

    # --- Department label ---
    st.markdown("<p class='dept-label-custom'>Department</p>", unsafe_allow_html=True)
    selected_department = st.selectbox("", DEPARTMENTS)

    # --- Instructions ---
    st.markdown("""
    <div class='instruction-box'>
      <p><strong>Instructions:</strong> 
      Click the microphone below, speak, and click again to stop. 
      Your audio will be uploaded and transcribed automatically. 
      Then choose the best transcript.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Audio Recorder ---
    audio_data = st_audiorec()

    # If the user recorded audio, show the length
    if audio_data:
        st.write(f"Received {len(audio_data)} bytes of audio data.")
    else:
        st.info("No audio recorded yet. Please click the microphone to record.")

    # If audio is available, proceed
    if audio_data:
        # 1) Upload audio to S3
        with st.spinner("Uploading audio to S3..."):
            try:
                audio_url = upload_audio_to_s3(audio_data)
            except Exception as e:
                st.error(f"Error uploading to S3: {e}")
                return

        # 2) Transcribe in parallel
        with st.spinner("Transcribing..."):
            deepgram_text, whisper_text, assemblyai_text = transcribe_all_in_parallel(audio_data)

        st.success("Transcription completed!")

        # 3) Display transcripts
        st.subheader("Transcribed Results")
        tabs = st.tabs(["Deepgram", "Whisper", "AssemblyAI"])
        with tabs[0]:
            st.write(deepgram_text if deepgram_text else "No transcript")
        with tabs[1]:
            st.write(whisper_text if whisper_text else "No transcript")
        with tabs[2]:
            st.write(assemblyai_text if assemblyai_text else "No transcript")

        # 4) Choose best transcript
        st.subheader("Choose the most accurate transcript")
        choice = st.radio("", ["Deepgram", "Whisper", "AssemblyAI"], index=0)

        # 5) Save to database
        if st.button("Save Transcript"):
            # Make sure at least one transcript is non-empty
            if not (deepgram_text.strip() or whisper_text.strip() or assemblyai_text.strip()):
                st.warning("All transcripts are empty. Cannot save.")
            else:
                save_transcripts_to_postgres(
                    dept=selected_department,
                    audio_url=audio_url,
                    deepgram_text=deepgram_text,
                    whisper_text=whisper_text,
                    assemblyai_text=assemblyai_text,
                    chosen_engine=choice
                )
                st.success("Record saved to PostgreSQL successfully!")


# ----------------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
