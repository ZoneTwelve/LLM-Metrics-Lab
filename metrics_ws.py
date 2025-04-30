import asyncio
import time
import json
from datetime import datetime
import time
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
import math
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import os
from dotenv import load_dotenv
import uuid
from utils.url import normalize_url
from datasets import load_dataset
import websockets
import httpx

from utils.logger_config import setup_logger
from pathlib import Path

from config.settings import (
    LOG_FILE_DIR,
)
from visualize import APIMetricsVisualizer

logger = setup_logger(__name__)
logger.info("WebSocket server started")

class Template(str):
    pass

class Conversation(str):
    pass

load_dotenv()

runtime_uuid = None

# Suppress SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

questions = []
count_id = 0
# Global monitor instance
monitor = None
monitor_task = None
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
    def __init__(self, model: str, api_url: str, api_key: str, max_concurrent: int = 5, columns: int = 3, log_file: str = "api_monitor.jsonl", plot_file: str = "api_metrics.png", output_dir: str = None):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.columns = columns
        self.log_file = log_file
        self.plot_file = plot_file
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
        self._stop_requested = False
        
        # Initialize log file
        with open(Path(self.output_dir, self.log_file).resolve(), 'w') as f:
            f.write('')
            f.close()
        
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
        
    async def generate_status_table(self, websocket):
        table = Table(
            title="API Throughput Monitor",
            box=box.ROUNDED,
            title_style="bold magenta",
            header_style="bold cyan",
        )

        for i in range(self.columns):
            table.add_column(f"Session Group {i+1}", justify="left")

        sessions_data = {}
        
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
                        sessions_data[session_id] = info
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
            if self.total_requests:
                success_rate = round(self.successful_requests/self.total_requests * 100, 2)
            else:
                success_rate = 0
            
            summary_stats_to_send = {
                "time": elapsed_time,
                "active": self.active_sessions,
                "total": self.total_requests,
                "success": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": success_rate,
                "chars_per_sec": chars_per_sec,
                "total_chars": total_chars,
                "total_chunks": total_chunks
            }
            
            await websocket.send(json.dumps({
                "status": "stats_update",
                "data": {
                    "sessions": sessions_data,
                    "dashboard": summary_stats_to_send
                }
            }))

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
            
            with open(Path(self.output_dir, self.log_file).resolve(), 'a') as f:
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
    

    async def make_request(self, session_id):
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
            
            # Make request with SSL verification disabled
            async with httpx.AsyncClient(verify=False, timeout=180.0) as client:
                async with client.stream("POST", f"{self.api_url}/chat/completions", headers=headers, json=payload) as response:
                    logger.debug(f"RESPONSE STATUS: {response.status_code}")
                    payload_record = FileHandler(f"{self.output_dir}/in_{runtime_uuid}_{session_id}.json", "w", True)
                    output_record = FileHandler(f"{self.output_dir}/out_{runtime_uuid}_{session_id}.json", "w", True)

                    payload_record.write(json.dumps(payload))
                    payload_record.close()

                    async for line in response.aiter_lines():
                        if line:
                            data = self.process_stream_info(line)
                            if data is None:
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
    
    async def run(self, websocket, duration=10):
        """Async version of run for WebSocket integration"""
        self.duration = duration
        end_time = time.time() + duration
        session_id = 0
        self.running = True
        self._stop_requested = False
        
        logger.info(f"🧪 Starting run loop for {duration}s with max_concurrent={self.max_concurrent}")
        
        with Live(
            await self.generate_status_table(websocket),
            refresh_per_second=4,
            vertical_overflow="visible",
            auto_refresh=True
        ) as live:
            try: 
                while time.time() < end_time and self.running and not self._stop_requested:
                    current_time = time.time()

                    if current_time - self.last_log_time >= 1.0:
                        await self.log_status()

                    if self.active_sessions < self.max_concurrent:
                        session_id += 1
                        async with self.lock:
                            self.active_sessions += 1
                        asyncio.create_task(self.make_request(session_id))
                    if self.should_update_display():
                        live.update(await self.generate_status_table(websocket))

                    await asyncio.sleep(0.1)
            finally:
                # Send log file to frontend
                file_info = {
                    "status": "file",
                    "fileName": self.log_file,
                    "fileUrl": f"/downloads/{self.log_file}"
                }
                try:
                    await websocket.send(json.dumps(file_info))
                    logger.info(f"📦 Log file info sent to frontend.")
                except Exception as e:
                    logger.error(f"❌ Failed to send log file info to frontend: {e}")
                # Generate charts
                try:
                    log_file_path = Path(self.output_dir, self.log_file).resolve()
                    plot_file_path = Path(self.output_dir, self.plot_file).resolve()
                    generate_visualization(log_file_path, plot_file_path)
                    logger.info(f"📊 Generating visualization from {log_file_path} to {plot_file_path}")
                    plot_info = {
                        "status": "file",
                        "fileName": self.plot_file,
                        "fileUrl": f"/downloads/{self.plot_file}"
                    }
                    await websocket.send(json.dumps(plot_info))
                    logger.info("📈 Visualization generated successfully.")
                except Exception as e:
                    logger.error(f"❌ Failed to generate visualization: {e}")
                # Clean up states
                self.running = False
                self._stop_requested = False
                logger.info("🛑 run() has ended (timeout or stopped).")
                logger.info(f"File Path: {file_info}")

    async def stop_monitor(self):
        self.running = False
        self._stop_requested = True


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
                if isinstance(row[key], dict) or isinstance(row[key], list):
                    messages = row[key]
                else:
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

async def websocket_handler(websocket):
    global monitor, monitor_task, connected_clients, count_id
    
    logger.info(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                print("!!! Failed to parse message:", e)
                continue

            # 處理 "start" 命令
            if data.get("command") == "start":
                runtime_uuid = str(uuid.uuid4()).replace("-", "")
                if monitor and monitor.running:
                    await websocket.send(json.dumps({"status": "error", "message": "Monitor already running"}))
                    logger.info(f"Monitor already running: {monitor.sessions}")
                else:
                    params = data.get("params", {})
                    model = params.get('model', os.getenv('MODEL', 'gpt-3.5-turbo'))
                    api_url = normalize_url(params.get('api_url', os.environ.get('API_URL')))
                    max_concurrent = int(params.get('max_concurrent', 5))
                    columns = int(params.get('columns', 3))
                    # log_file = params.get('log_file', "api_monitor.jsonl")
                    # plot_file = params.get('plot_file', "api_metrics.png")
                    log_file = f"{runtime_uuid}.jsonl"
                    plot_file = f"{runtime_uuid}.png"
                    # output_dir = params.get('output_dir') # Dangerous, this might cause security issues like overwriting files or directories discovery
                    time_limit = int(params.get('time_limit', 10))
                    dataset_name = params.get('dataset', "tatsu-lab/alpaca") # Dangours, use with caution
                    template_str = params.get('template')
                    conversation_str = params.get('conversation')

                    # Load it from environment variables
                    output_dir = LOG_FILE_DIR

                    # Create directories if needed
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Set up the monitor
                    global questions
                    api_key = os.environ.get('OPENAI_API_KEY')
                    
                    # Load dataset
                    logger.info(f"Loading dataset '{dataset_name}' with template '{template_str}' or conversation '{conversation_str}'")
                    if template_str is not None and template_str != "":
                        questions = load_dataset_as_questions(dataset_name, Template(template_str))
                    elif conversation_str is not None and conversation_str != "":
                        questions = load_dataset_as_questions(dataset_name, Conversation(conversation_str))
                    else:
                        await websocket.send(json.dumps({"status": "error", "message": "Either template or conversation must be provided"}))
                        continue
                    
                    if monitor:
                        await monitor.stop_monitor()
                    monitor = APIThroughputMonitor(
                        model=model,
                        api_url=api_url,
                        api_key=api_key,
                        max_concurrent=max_concurrent,
                        columns=columns,
                        log_file=log_file,
                        plot_file=plot_file,
                        output_dir=output_dir
                    )
                    
                    # Start the monitor
                    logger.info("🚀 Starting API Throughput Monitor...")
                    await websocket.send(json.dumps({"status": "started", "message": "Monitor started"}))
                    
                    # Run the monitor in the background
                    monitor_task = asyncio.create_task(monitor.run(websocket, duration=time_limit))

            elif data.get("command") == "stop":
                if monitor and monitor.running:
                    await monitor.stop_monitor()
                    if monitor_task:
                        monitor_task.cancel()
                        try:
                            await monitor_task
                        except asyncio.CancelledError:
                            logger.info("Monitor task cancelled successfully.")

                    monitor = None
                    monitor_task = None
                    count_id = 0
                    await websocket.send(json.dumps({"status": "stopping", "message": "Monitor stopping"}))
                else:
                    await websocket.send(json.dumps({"status": "error", "message": "No monitor running"}))


    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {websocket.remote_address}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Monitor Task Done (after disconnect or error)")
        
async def monitor_cleaner():
    """Background task to clean up monitor after it finishes"""
    global monitor, monitor_task, count_id, connected_clients
    while True:
        if monitor_task is not None and monitor_task.done():
            monitor = None
            monitor_task = None
            count_id = 0
            logger.info("✅ Monitor task finished, cleaning up")
            
            # Send completion message to all connected clients
            for websocket in connected_clients:
                try:
                    completion_message = {
                        "status": "completed",
                        "message": "Benchmark run finished"
                    }
                    await websocket.send(json.dumps(completion_message))
                    logger.info(f"📡 Sent completion message to {websocket.remote_address}")
                except Exception as e:
                    logger.error(f"Error sending message to {websocket.remote_address}: {str(e)}")
                    
        await asyncio.sleep(0.5)
        
def generate_visualization(log_file, plot_file):
    visualizer = APIMetricsVisualizer(log_file)
    visualizer.create_visualization(plot_file)