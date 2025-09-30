import os
import uuid
import shutil
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from pydub.generators import Sine
from faster_whisper import WhisperModel
from moviepy import VideoFileClip, AudioFileClip


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


@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOADS_DIR, f"{unique_id}_{file.filename}")
    base_name, ext = os.path.splitext(file.filename)

    output_audio_path = os.path.join(PROCESSED_DIR, f"{unique_id}_censored.wav")
    output_video_path = os.path.join(PROCESSED_DIR, f"{unique_id}_censored.mp4")

    video = None
    censored_audio_clip = None
    final_video = None
    temp_audio_path = None

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Audio Processing ---
        if ext.lower() in [".mp3", ".wav", ".m4a"]:
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

            if words_to_censor:
                censored_audio = censor_audio_with_beep(input_path, words_to_censor)
                censored_audio.export(output_audio_path, format="wav")
            else:
                shutil.copy(input_path, output_audio_path)

            return {"processed_url": f"/processed/{os.path.basename(output_audio_path)}"}

        # --- Video Processing ---
        elif ext.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            video = VideoFileClip(input_path)

            temp_audio_path = os.path.join(UPLOADS_DIR, f"{unique_id}_temp_audio.wav")
            video.audio.write_audiofile(temp_audio_path, codec="pcm_s16le")

            segments, _ = model.transcribe(temp_audio_path, word_timestamps=True)

            words_to_censor = []
            for segment in segments:
                for word in segment.words:
                    cleaned = normalize_word(word.word)
                    if cleaned in ABUSIVE_WORDS:
                        print(f"🚨 Found abusive word: {word.word} -> {cleaned} at {word.start:.2f}s")
                        words_to_censor.append(
                            {"word": cleaned, "start": word.start, "end": word.end}
                        )

            if words_to_censor:
                censored_audio = censor_audio_with_beep(temp_audio_path, words_to_censor)
                censored_audio.export(output_audio_path, format="wav")

                censored_audio_clip = AudioFileClip(output_audio_path)
                final_video = video.with_audio(censored_audio_clip)  # ✅ fixed
                final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

                return {"processed_url": f"/processed/{os.path.basename(output_video_path)}"}
            else:
                shutil.copy(input_path, output_video_path)
                return {"processed_url": f"/processed/{os.path.basename(output_video_path)}"}

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload audio/video only.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ✅ Close clips to release file handles
        try:
            if video:
                video.close()
            if censored_audio_clip:
                censored_audio_clip.close()
            if final_video:
                final_video.close()
        except:
            pass

        # ✅ Remove temp files safely
        for path in [input_path, temp_audio_path, output_audio_path]:
            try:
                if path and os.path.exists(path) and "censored" not in path:  # don't delete final output
                    os.remove(path)
            except:
                pass
