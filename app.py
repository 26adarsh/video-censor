import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from pydub.generators import Sine
from faster_whisper import WhisperModel

# --- 1. INITIAL SETUP ---

# Initialize the FastAPI app
app = FastAPI()

# Setup for templates
templates = Jinja2Templates(directory="templates")

# Create directories for temporary files
UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Mount the 'processed' directory to serve the final videos
app.mount("/processed_videos", StaticFiles(directory=PROCESSED_DIR), name="processed_videos")

# Load the abusive words list from the text file
try:
    with open("abusive_words.txt", "r") as f:
        ABUSIVE_WORDS = {line.strip().lower() for line in f}
except FileNotFoundError:
    print("Warning: 'abusive_words.txt' not found. No words will be censored.")
    ABUSIVE_WORDS = set()

# Load the Whisper model
MODEL_SIZE = "base"
print(f"Loading Whisper model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Whisper model loaded.")


# --- 2. CORE LOGIC ---

def censor_audio_with_beep(original_audio_path: str, abusive_words_timestamps: list):
    """Overlays a beep sound over the detected abusive words."""
    audio = AudioSegment.from_file(original_audio_path)

    for word_info in abusive_words_timestamps:
        start_time_ms = int(word_info['start'] * 1000)
        end_time_ms = int(word_info['end'] * 1000)
        duration = end_time_ms - start_time_ms

        beep = Sine(1000).to_audio_segment(duration=duration, volume=-10)
        audio = audio.overlay(beep, position=start_time_ms)

    return audio


# --- 3. API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """The main endpoint to upload, process, and return a censored video."""
    unique_id = str(uuid.uuid4())
    video_extension = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOADS_DIR, f"{unique_id}{video_extension}")
    audio_path = os.path.join(UPLOADS_DIR, f"{unique_id}.wav")
    censored_audio_path = os.path.join(UPLOADS_DIR, f"{unique_id}_censored.wav")
    output_video_path = os.path.join(PROCESSED_DIR, f"{unique_id}_censored.mp4")

    try:
        # 1. Save uploaded video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract audio
        print(f"Extracting audio from {video_path}...")
        os.system(f'ffmpeg -i "{video_path}" -vn -ar 16000 -ac 1 -y "{audio_path}"')

        # 3. Transcribe and find words to censor
        print("Transcribing audio...")
        segments, _ = model.transcribe(audio_path, word_timestamps=True)

        words_to_censor = []
        for segment in segments:
            for word in segment.words:
                if word.word.strip().lower() in ABUSIVE_WORDS:
                    print(f"Found word: '{word.word}' at {word.start:.2f}s")
                    words_to_censor.append({'word': word.word, 'start': word.start, 'end': word.end})

        # 4. Process and merge if necessary
        if words_to_censor:
            print("Censoring audio...")
            censored_audio = censor_audio_with_beep(audio_path, words_to_censor)
            censored_audio.export(censored_audio_path, format="wav")

            print("Merging censored audio back into video...")
            os.system(f'ffmpeg -i "{video_path}" -i "{censored_audio_path}" -c:v copy -map 0:v:0 -map 1:a:0 -y "{output_video_path}"')
        else:
            print("No abusive words found. Using original video.")
            shutil.copy(video_path, output_video_path)

        # 5. Return the URL to the processed video
        video_url = f"/processed_videos/{os.path.basename(output_video_path)}"
        return {"processed_video_url": video_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 6. Clean up temporary files
        print("Cleaning up temporary files...")
        for path in [video_path, audio_path, censored_audio_path]:
            if os.path.exists(path):
                os.remove(path)