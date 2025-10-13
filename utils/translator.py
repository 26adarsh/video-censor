# utils/translator.py
from googletrans import Translator

translator = Translator()

def translate_to_en(text: str, src_lang: str) -> str:
    """
    Translate a word/phrase to English using googletrans.
    - src_lang: 'hi' for Hindi, 'mr' for Marathi
    """
    try:
        translated = translator.translate(text, src=src_lang, dest="en")
        return translated.text.lower()
    except Exception as e:
        print(f"⚠️ Translation failed for '{text}': {e}")
        return text.lower()
