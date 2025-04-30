from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Assign static dir
STATIC_DIR = Path("interface").resolve() # TODO: check this implementation in the future
LOG_FILE_DIR = Path(os.getenv("LOG_FILE_DIR", '.outputs')).resolve()

HOST = os.getenv("HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", 8765))
HTTP_PORT = int(os.getenv("HTTP_PORT", 8000))