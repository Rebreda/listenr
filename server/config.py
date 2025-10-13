import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    UPLOAD_FOLDER = os.environ.get("LISTENR_STORAGE", os.path.join(BASE_DIR, "../uploads"))
    ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg"}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    DEBUG = os.environ.get("DEBUG", "true").lower() == "true"

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
