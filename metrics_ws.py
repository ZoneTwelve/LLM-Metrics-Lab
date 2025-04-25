import requests
import asyncio
import threading
import time
import json
from datetime import datetime
import time
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import math
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import os
from dotenv import load_dotenv
import uuid
from utils.url import normalize_url
from datasets import load_dataset
import websockets
from aiohttp import web
import httpx

import logging

class Template(str):
    pass

class Conversation(str):
    pass

load_dotenv()

# Configure logging level from environment variable
# (Console/File) Log levels
console_log_level = os.getenv("CLL", "info").upper()
file_log_level = os.getenv("FLL", "").upper()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('Metrics')
# logger.setLevel(getattr(logging, console_log_level, logging.DEBUG))

runtime_uuid = str(uuid.uuid4()).replace("-", "")

if "" != console_log_level:
    # Add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)

    # Define a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

if "" != file_log_level:
    # Add a file handler
    file_handler = logging.FileHandler("metrics.log")
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Suppress SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

questions = []
count_id = 0
connected_clients = set()

class FileHandler:
    def __init__(self, filename: str, mode: str, virtual: bool = False):
        self.filename = filename
        self.file = open(filename, mode) if not virtual else None

    def write(self, data):
        if self.file:
            self.file.write(data)

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class APIThroughputMonitor:
    def __init__(self, model: str, api_url: str, api_key: str, max_concurrent: int = 5, columns: int = 3, log_file: str = "api_monitor.jsonl", output_dir: str = None):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.columns = columns
        self.log_file = log_file
        self.sessions = {}
        self.lock = asyncio.Lock()
        self.console = Console()
        self.active_sessions = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.prev_total_chars = 0
        self.last_update_time = self.start_time
        self.update_interval = 0.25
        self.output_dir = output_dir
        self.running = True
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write('')
    def get_session_status(self, session_id, info):
        status_style = {
            "Starting": "yellow",
            "Processing": "blue",
            "Completed": "green",
            "Failed": "red"
        }.get(info["status"], "white")

        return (
            f"{session_id:3d} | "
            f"[{status_style}]{info['status']:10}[/{status_style}] | "
            f"Time: {info['response_time'] or '-':8} | "
            f"Chars: {info['total_chars']:5} | "
            f"Chunks: {info['chunks_received']:3}"
        )
        
    async def generate_status_table(self):
        table = Table(
            title="API Throughput Monitor",
            box=box.ROUNDED,
            title_style="bold magenta",
            header_style="bold cyan",
        )

        for i in range(self.columns):
            table.add_column(f"Session Group {i+1}", justify="left")

        async with self.lock:
            sorted_sessions = sorted(self.sessions.items(), key=lambda x: int(x[0]))
            num_sessions = len(sorted_sessions)
            num_rows = math.ceil(num_sessions / self.columns)

            for row_idx in range(num_rows):
                row_data = []
                for col_idx in range(self.columns):
                    session_idx = row_idx * self.columns + col_idx
                    if session_idx < len(sorted_sessions):
                        session_id, info = sorted_sessions[session_idx]
                        row_data.append(self.get_session_status(session_id, info))
                    else:
                        row_data.append("")
                table.add_row(*row_data)

            elapsed_time = time.time() - self.start_time
            total_chars = sum(s["total_chars"] for s in self.sessions.values())
            total_chunks = sum(s["chunks_received"] for s in self.sessions.values())
            chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0

            table.add_section()
            stats_summary = (
                f"[bold cyan]Summary Stats:[/bold cyan]\n"
                f"Time: {elapsed_time:.1f}s \n"
                f"Active: {self.active_sessions} | "
                f"Total: {self.total_requests} | "
                f"Success: {self.successful_requests} | "
                f"Failed: {self.failed_requests}\n"
                f"Chars/s: {chars_per_sec:.1f} | "
                f"Total Chars: {total_chars} | "
                f"Total Chunks: {total_chunks}"
            )
            table.add_row(stats_summary)

        return table
    
    async def log_status(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        async with self.lock:
            total_chars = sum(session["total_chars"] for session in self.sessions.values())
            chars_per_second = (total_chars - self.prev_total_chars) / (current_time - self.last_log_time)
            active_sessions = len([s for s in self.sessions.values() if s["status"] in ["Starting", "Processing"]])
            completed_sessions = len([s for s in self.sessions.values() if s["status"] == "Completed"])

            tokens_latency = [self.sessions[id]['tokens_latency'] for id in self.sessions]
            tokens_amount = [self.sessions[id]['tokens_amount'] for id in self.sessions]
            ftls = []
            try:
                ftls = [self.sessions[id]['first_token_latency'] for id in self.sessions]
                logger.debug(ftls)
            except KeyError as e:
                logger.error(e)
                ftls = []
            
            for id in self.sessions:
                self.sessions[id]['tokens_latency'] = []
                self.sessions[id]['tokens_amount'] = []

            status = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "total_chars": total_chars,
                "chars_per_second": round(chars_per_second, 2),
                "active_sessions": active_sessions,
                "completed_sessions": completed_sessions,
                "total_sessions": len(self.sessions),
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "tokens_latency": tokens_latency,
                "tokens_amount": tokens_amount,
                "first_token_latencies": ftls,
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(status) + '\n')
            
            self.prev_total_chars = total_chars
            self.last_log_time = current_time
            
            # Return status for potential use
            return status
    
    def process_stream_line(self, line):
        try:
            # Decode the line from bytes to string if necessary
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            # Remove the "data: " prefix if it exists
            logger.debug(f"Line: {line}")
            if line.startswith('data: '):
                line = line[6:]

            # Handle stream completion marker
            if line.strip() == '[DONE]':
                return None

            # Parse the JSON content
            data = json.loads(line)
            
            # Extract the content from the response structure
            if 'choices' in data and len(data['choices']) > 0:
                if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                    return data['choices'][0]['delta']['content']

            return None
        except json.JSONDecodeError:
            logger.error("<<< JSON parsing error >>")
            return None
        except Exception as e:
            logger.error(f"Error processing line: {str(e)}")
            return None

    def process_stream_info(self, line):
        try:
            # Decode the line from bytes to string if necessary
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            # Remove the "data: " prefix if it exists
            data_key = 'data: '
            logger.debug(f"Line: {line}")
            if line.startswith(data_key):
                line = line[len(data_key):]

            if line.strip() == '[DONE]':
                return None

            data = json.loads(line)
            elapsed_time = time.time() - self.start_time
            return {"data": data, "timestamp": time.time(), "in-time": self.duration > elapsed_time}
        except json.JSONDecodeError:
            logger.error("<<< JSON parsing error >>")
            logger.debug(f"Error processing line: {line}")
            return None
        except Exception as e:
            logger.error(f"Error processing line: {str(e)}")
            logger.debug(f"Error processing line: {line}")
            return None
    
    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        total_chars = sum(s.get("total_chars", 0) for s in self.sessions.values())
        total_chunks = sum(s.get("chunks_received", 0) for s in self.sessions.values())
        chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0

        session_data = []
        for session_id, s in self.sessions.items():
            session_data.append({
                "id": session_id,
                "status": s.get("status", "Unknown"),
                "startTime": datetime.fromtimestamp(s.get("start_time")).isoformat() if s.get("start_time") else None,
                "chars": s.get("total_chars", 0),
                "chunks": s.get("chunks_received", 0),
            })

        return {
            "status": "ok",
            "data": {
                "sessions": session_data,
                "summary": {
                    "elapsed_time": round(elapsed_time, 2),
                    "active": self.active_sessions,
                    "total": self.total_requests,
                    "success": self.successful_requests,
                    "failed": self.failed_requests,
                    "chars_per_sec": round(chars_per_sec, 2),
                    "total_chars": total_chars,
                    "total_chunks": total_chunks,
                }
            }
        }

    async def make_request(self, session_id):
        logger.info(f"👷 make_request START: session_id={session_id}")
        global count_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = questions[session_id % len(questions)]
        logger.debug(f"MESSAGE: {messages}")
        payload = {
            "model": self.model,
            "stream": True,
            "messages": messages
        }
        count_id += 1

        try:
            async with self.lock:
                self.sessions[session_id] = {
                    "status": "Starting",
                    "start_time": time.time(),
                    "response_time": None,
                    "error": None,
                    "total_chars": 0,
                    "chunks_received": 0,
                    "tokens_latency": [],
                    "tokens_amount": [],
                    "first_token_latency": -1,
                }

            start_time = time.time()
            next_token_time = start_time
            
            logger.info(f"Sending request to {self.api_url}/chat/completions with payload: {json.dumps(payload)}")

            # Make request with SSL verification disabled
            async with httpx.AsyncClient(verify=False, timeout=180.0) as client:
                async with client.stream("POST", f"{self.api_url}/chat/completions", headers=headers, json=payload) as response:
                    logger.debug(f"RESPONSE STATUS: {response.status_code}")
                    payload_record = FileHandler(f"{self.output_dir}/in_{runtime_uuid}_{session_id}.json", "w", self.output_dir is None)
                    output_record = FileHandler(f"{self.output_dir}/out_{runtime_uuid}_{session_id}.json", "w", self.output_dir is None)

                    payload_record.write(json.dumps(payload))
                    payload_record.close()

                    async for line in response.aiter_lines():
                        if line:
                            data = self.process_stream_info(line)
                            if data is None:
                                logger.info(f"Processing finished for session {session_id}")
                                break
                            output_record.write(json.dumps(data) + "\n")

                            content = data["data"]["choices"][0]["delta"].get("content", "")
                            async with self.lock:
                                latency = round(time.time() - next_token_time, 5)
                                self.sessions[session_id]["status"] = "Processing"
                                self.sessions[session_id]["chunks_received"] += 1
                                self.sessions[session_id]["total_chars"] += len(content)
                                self.sessions[session_id]["tokens_amount"].append(len(content))
                                self.sessions[session_id]["tokens_latency"].append(latency)
                                if self.sessions[session_id]["first_token_latency"] == -1:
                                    self.sessions[session_id]["first_token_latency"] = latency
                                next_token_time = time.time()

                    output_record.close()

            response_time = time.time() - start_time
            async with self.lock:
                self.sessions[session_id].update({
                    "status": "Completed",
                    "response_time": f"{response_time:.2f}s",
                    "error": None
                })
                self.successful_requests += 1
            logger.info(f"✅ make_request END: session_id={session_id}")
        except Exception as e:
            async with self.lock:
                logger.error(f"Error in session {session_id}: {str(e)}")
                self.sessions[session_id].update({
                    "status": "Failed",
                    "error": str(e),
                    "response_time": "N/A"
                })
                self.failed_requests += 1

        finally:
            async with self.lock:
                self.total_requests += 1
                self.active_sessions -= 1
                
    

    def should_update_display(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    async def run(self, duration=10):
        """Async version of run for WebSocket integration"""
        self.duration = duration
        end_time = time.time() + duration
        session_id = 0
        self.running = True
        
        logger.info(f"🧪 Starting run loop for {duration}s with max_concurrent={self.max_concurrent}")
        
        with Live(
            await self.generate_status_table(),
            refresh_per_second=4,
            vertical_overflow="visible",
            auto_refresh=True
        ) as live:
            while time.time() < end_time and self.running:
                logger.info(f"end_time: {end_time}, time: {time.time()}")
                logger.info(f"[run loop] active: {self.active_sessions}, max: {self.max_concurrent}")
                current_time = time.time()

                if current_time - self.last_log_time >= 1.0:
                    await self.log_status()

                logger.info(f"[run loop] active: {self.active_sessions}, max: {self.max_concurrent}")
                if self.active_sessions < self.max_concurrent:
                    session_id += 1
                    logger.info(f"🚀 Launching make_request for session {session_id}")
                    async with self.lock:
                        self.active_sessions += 1
                    asyncio.create_task(self.make_request(session_id))

                await asyncio.sleep(0.1)

        logger.info("🛑 run() has ended due to time_limit.")
        self.running = False
        await asyncio.sleep(1)

def load_dataset_as_questions(dataset_name: str, key: Template | Conversation):
    dataset = load_dataset(dataset_name)['train']
    ret = []
    if isinstance(key, Template):
        ret = []
        for row in dataset:
            conv = [
                {"role": "user", "content": key.format(**row)},
            ]
            ret.append(conv)
    elif isinstance(key, Conversation):
        for row in dataset:
            try:
                messages = json.loads(row[key])
                for turn in messages:
                    if 'role' in turn and 'content' in turn and isinstance(turn['role'], str) and isinstance(turn['content'], str):
                        ret.append(messages)
                    else:
                        raise ValueError(f"Invalid conversation context")
            except json.JSONDecodeError as e:
                raise ValueError(f"Can not load columns '{key}' as Conversation template")
    else:
        ret = None
    return ret

# Global monitor instance
monitor = None

async def websocket_handler(websocket):
    global monitor, connected_clients
    
    logger.info(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            logger.info("<<< Received message >>>")  # 這是 debug print
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                print("!!! Failed to parse message:", e)
                continue
            
            logger.info("[RECV] Raw message:", message)
            logger.info("[RECV] Parsed data:", data)

            if data.get("command") != "start":
                await websocket.send(json.dumps({"error": "Unknown command"}))
                continue

            # 處理 "start" 命令
            if data.get("command") == "start":
                if monitor and monitor.running:
                        await websocket.send(json.dumps({"status": "error", "message": "Monitor already running"}))
                else:
                    params = data.get("params", {})
                    print("[INFO] Received params:", params)
                    model = params.get('model', os.getenv('MODEL', 'gpt-3.5-turbo'))
                    api_url = normalize_url(params.get('api_url', os.environ.get('API_URL')))
                    max_concurrent = int(params.get('max_concurrent', 5))
                    columns = int(params.get('columns', 3))
                    log_file = params.get('log_file', "api_monitor.jsonl")
                    output_dir = params.get('output_dir')
                    time_limit = int(params.get('time_limit', 10))
                    dataset_name = params.get('dataset', "tatsu-lab/alpaca")
                    template_str = params.get('template')
                    conversation_str = params.get('conversation')

                    # Create directories if needed
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Set up the monitor
                    global questions
                    api_key = os.environ.get('OPENAI_API_KEY')
                    
                    # Load dataset
                    if template_str is not None:
                        questions = load_dataset_as_questions(dataset_name, Template(template_str))
                    elif conversation_str is not None:
                        questions = load_dataset_as_questions(dataset_name, Conversation(conversation_str))
                    else:
                        await websocket.send(json.dumps({"status": "error", "message": "Either template or conversation must be provided"}))
                        continue
                    
                    monitor = APIThroughputMonitor(
                        model=model,
                        api_url=api_url,
                        api_key=api_key,
                        max_concurrent=max_concurrent,
                        columns=columns,
                        log_file=log_file,
                        output_dir=output_dir,
                    )
                    
                    # Start the monitor
                    logger.info("🚀 Starting API Throughput Monitor...")
                    await websocket.send(json.dumps({"status": "started", "message": "Monitor started"}))
                    
                    # Run the monitor in the background
                    asyncio.create_task(monitor.run(duration=time_limit))
                    
                    logger.info(f"RUN INFO: {monitor.running}")
                    
                    # Send sessions status to frontend
                    while monitor.running:
                        logger.info(f"RUNNING")
                        await asyncio.sleep(1)
                        stats = monitor.get_stats()
                        try:
                            logger.info(f"Sending periodic stats to frontend.")
                            await websocket.send(json.dumps(stats))
                        except Exception as e:
                            logger.error(f"Error sending stats to frontend: {e}")
                            break

                    # Send final status
                    if not monitor.running:
                        logger.info(f"Sending final stats to frontend.")
                        await websocket.send(json.dumps({
                            "status": "completed",
                            "message": "Monitor completed",
                            "data": monitor.get_stats()
                        }))

            # 處理 "confirm" 命令
            elif data.get("command") == "confirm":
                print("[INFO] Received confirmation from frontend:", data)
                confirmation_status = {
                    "status": "received",
                    "message": "Task received by frontend"
                }

                # 可以選擇發送回應給前端
                await websocket.send(json.dumps({"status": "acknowledged", "message": confirmation_status}))
                print("[INFO] Sent acknowledgment to frontend.")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {websocket.remote_address}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        connected_clients.discard(websocket)

async def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8765))
    server = await websockets.serve(websocket_handler, host, port)
    print(f"WebSocket server running at ws://{host}:{port}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
