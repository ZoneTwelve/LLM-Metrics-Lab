import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import websockets

from metrics_ws import websocket_handler, monitor_cleaner
from utils.logger_config import setup_logger

from config.settings import (
    LOG_FILE_DIR,
    STATIC_DIR,
    HOST,
    WS_PORT,
    HTTP_PORT,
)

# --- Set logger ---
logger = setup_logger(__name__)

# --- FastAPI app ---
app = FastAPI()

try:
    app.mount("/downloads", StaticFiles(directory=LOG_FILE_DIR), name="downloads")
except Exception as e:
    logger.error(f"Error mounting downloads directory: {e}, ")
    raise RuntimeError(f"Please create the directory '{LOG_FILE_DIR}' manually.")

# get web interface
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "preview.html"
    return index_file.read_text(encoding="utf-8")

# Activate WebSocket and FastAPI
async def main():
    host = HOST
    ws_port = WS_PORT
    http_port = HTTP_PORT

    # Activate WebSocket server
    ws_server = await websockets.serve(websocket_handler, host, ws_port)
    logger.info(f"✅ WebSocket server running at ws://{host}:{ws_port}")

    # Activate FastAPI (HTTP) server
    config = uvicorn.Config(app, host=host, port=http_port, log_level="info")
    http_server = uvicorn.Server(config)
    asyncio.create_task(http_server.serve())
    logger.info(f"✅ FastAPI server running at http://{host}:{http_port}")

    # Activate cleaner
    asyncio.create_task(monitor_cleaner())

    await ws_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
