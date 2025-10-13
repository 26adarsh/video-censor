# utils/file_utils.py
import os
import uuid

def generate_unique_filename(original_name: str, folder: str, suffix: str = "") -> str:
    unique_id = str(uuid.uuid4())
    name, ext = os.path.splitext(original_name)
    filename = f"{unique_id}_{name}{suffix}{ext}"
    return os.path.join(folder, filename)

def safe_remove(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"⚠️ Could not delete {file_path}: {e}")
