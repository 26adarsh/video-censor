import os
import uuid
import shutil
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from pydub.generators import Sine
from faster_whisper import WhisperModel

# --- Setup ---
app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOADS_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")

# Load abusive words
ABUSIVE_WORDS = set()
try:
    with open("abusive_words.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            if "|" in line:
                ABUSIVE_WORDS.update(line.split("|"))
            else:
                ABUSIVE_WORDS.add(line)
    print(f"✅ Loaded {len(ABUSIVE_WORDS)} abusive words")
except FileNotFoundError:
    print("⚠️ abusive_words.txt not found. No censoring will happen.")


# Normalize words before checking
def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())  # remove punctuation & lowercase


# Load Whisper
print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper loaded ✅")


# --- Core function ---
def censor_audio_with_beep(original_audio_path: str, abusive_words_timestamps: list):
    audio = AudioSegment.from_file(original_audio_path)

    for word_info in abusive_words_timestamps:
        start_time_ms = int(word_info["start"] * 1000)
        end_time_ms = int(word_info["end"] * 1000)
        duration = max(200, end_time_ms - start_time_ms)  # at least 200ms

        beep = Sine(1000).to_audio_segment(duration=duration, volume=-5)
        # Replace segment with beep instead of overlay
        audio = audio[:start_time_ms] + beep + audio[end_time_ms:]

    return audio


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOADS_DIR, f"{unique_id}_{file.filename}")
    output_path = os.path.join(PROCESSED_DIR, f"{unique_id}_censored.wav")

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe with Whisper
        segments, _ = model.transcribe(input_path, word_timestamps=True)

        words_to_censor = []
        for segment in segments:
            for word in segment.words:
                cleaned = normalize_word(word.word)
                if cleaned in ABUSIVE_WORDS:
                    print(f"🚨 Found abusive word: {word.word} -> {cleaned} at {word.start:.2f}s")
                    words_to_censor.append(
                        {"word": cleaned, "start": word.start, "end": word.end}
                    )

        # Apply censoring
        if words_to_censor:
            censored_audio = censor_audio_with_beep(input_path, words_to_censor)
            censored_audio.export(output_path, format="wav")
        else:
            shutil.copy(input_path, output_path)

        return {"processed_url": f"/processed/{os.path.basename(output_path)}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
