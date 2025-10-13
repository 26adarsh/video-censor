# utils/censor.py
from pydub import AudioSegment
from pydub.generators import Sine
from .normalize import normalize_word

def censor_audio_with_beep(original_audio_path: str, abusive_words_timestamps: list, lang: str = "en") -> AudioSegment:
    """
    Censor audio by replacing abusive words with beep.
    - abusive_words_timestamps: list of dicts: {"word":..., "start":..., "end":...}
    - lang: language code for normalization
    """
    audio = AudioSegment.from_file(original_audio_path)

    for word_info in abusive_words_timestamps:
        start_time_ms = int(word_info["start"] * 1000)
        end_time_ms = int(word_info["end"] * 1000)
        duration = max(200, end_time_ms - start_time_ms)  # at least 200ms

        beep = Sine(1000).to_audio_segment(duration=duration, volume=-5)
        # Replace segment with beep instead of overlay
        audio = audio[:start_time_ms] + beep + audio[end_time_ms:]

    return audio

def find_abusive_words(segments, abusive_words_set, lang="en"):
    """
    Returns a list of abusive words with timestamps.
    segments: Whisper transcription segments
    abusive_words_set: set of normalized abusive words in English
    """
    words_to_censor = []
    for segment in segments:
        for word in segment.words:
            normalized = normalize_word(word.word, lang)
            if normalized in abusive_words_set:
                words_to_censor.append(
                    {"word": word.word, "start": word.start, "end": word.end}
                )
    return words_to_censor
