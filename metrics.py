import asyncio
import os
import json
from datetime import datetime
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from rich.live import Live
from dotenv import load_dotenv
from datasets import load_dataset
import tqdm
import threading

from utils.utils import OpenAIPayload, OpenAIAPIRequest, openai_compatible_request
from utils.monitor import setup_logging, FileHandler, generate_status_table

load_dotenv()

questions = []
count_id = 0

# TODO: move APIThroughputMonitor to a separate module
# TODO: Refactor APIThroughputMonitor and create a BaseMonitor class

class APIThroughputMonitor:
    def __init__(self, model: str, api_url: str, api_key: str, max_concurrent: int = 5, 
                 columns: int = 3, log_file: str = "api_monitor.jsonl", output_dir: str = None):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.columns = columns
        self.log_file = log_file
        self.output_dir = output_dir
        self.sessions = {}
        self.lock = threading.Lock()
        self.active_sessions = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.prev_total_chars = 0
        self.last_update_time = self.start_time
        self.update_interval = 0.25  # Screen update interval in seconds
        self.logger = setup_logging(
            console_log_level=os.getenv("CLL", "INFO"),
            file_log_level=os.getenv("FLL", ""),
            log_file=log_file
        )
        self.runtime_uuid = str(uuid.uuid4()).replace("-", "")
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write('')

    async def make_request(self, session_id: int):
        """
        Make an asynchronous streaming request using utils.py's openai_compatible_request.
        """
        global count_id
        self.logger.debug(f"Making request for session {session_id}")
        
        payload = OpenAIPayload(
            model=self.model,
            messages=[{"role": "user", "content": questions[count_id % len(questions)]}],
            stream=True
        )
        request = OpenAIAPIRequest(
            api_url=f"{self.api_url}/chat/completions",
            api_key=self.api_key,
            payload=payload
        )
        count_id += 1
        
        try:
            with self.lock:
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
            
            # Record payload and output
            payload_record = FileHandler(
                f"{self.output_dir}/in_{self.runtime_uuid}_{session_id}.json",
                "w",
                virtual=self.output_dir is None
            )
            output_record = FileHandler(
                f"{self.output_dir}/out_{self.runtime_uuid}_{session_id}.json",
                "w",
                virtual=self.output_dir is None
            )
            payload_record.write(json.dumps(payload.to_openai_dict()))
            payload_record.close()
            
            async for chunk in await openai_compatible_request(request):
                content = chunk.chunk.choices[0].delta.content if chunk.chunk.choices[0].delta else ""
                if content is None:
                    break
                with self.lock:
                    latency = round(time.time() - next_token_time, 5)
                    self.sessions[session_id]["status"] = "Processing"
                    self.sessions[session_id]["chunks_received"] += 1
                    self.sessions[session_id]["total_chars"] += len(content)
                    self.sessions[session_id]["tokens_amount"].append(len(content))
                    self.sessions[session_id]["tokens_latency"].append(latency)
                    if self.sessions[session_id]["first_token_latency"] == -1:
                        self.sessions[session_id]["first_token_latency"] = latency
                    next_token_time = time.time()
                
                output_record.write(json.dumps({
                    "data": chunk.chunk.dict(),
                    "timestamp": chunk.timestamp,
                    "in-time": self.duration > (time.time() - self.start_time)
                }) + "\n")
            
            output_record.close()
            response_time = time.time() - start_time
            
            with self.lock:
                self.sessions[session_id].update({
                    "status": "Completed",
                    "response_time": f"{response_time:.2f}s",
                    "error": None
                })
                self.successful_requests += 1
        
        except Exception as e:
            with self.lock:
                self.logger.error(f"Error in session {session_id}: {str(e)}")
                self.sessions[session_id].update({
                    "status": "Failed",
                    "error": str(e),
                    "response_time": "N/A"
                })
                self.failed_requests += 1
        
        finally:
            with self.lock:
                self.total_requests += 1
                self.active_sessions -= 1

    def log_status(self):
        """
        Log the current status to the log file.
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        with self.lock:
            total_chars = sum(session["total_chars"] for session in self.sessions.values())
            chars_per_second = (total_chars - self.prev_total_chars) / (current_time - self.last_log_time)
            active_sessions = len([s for s in self.sessions.values() if s["status"] in ["Starting", "Processing"]])
            completed_sessions = len([s for s in self.sessions.values() if s["status"] == "Completed"])
            
            tokens_latency = [self.sessions[id]["tokens_latency"] for id in self.sessions]
            tokens_amount = [self.sessions[id]["tokens_amount"] for id in self.sessions]
            first_token_latencies = [self.sessions[id]["first_token_latency"] for id in self.sessions]
            
            for id in self.sessions:
                self.sessions[id]["tokens_latency"] = []
                self.sessions[id]["tokens_amount"] = []
            
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
                "first_token_latencies": first_token_latencies,
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(status) + '\n')
            
            self.prev_total_chars = total_chars
            self.last_log_time = current_time

    def should_update_display(self):
        """
        Determine if the display should be updated based on the update interval.
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    async def run(self, duration: int = 60):
        """
        Run the throughput monitor for the specified duration.
        """
        with Live(
            generate_status_table(self.sessions, self.columns, self.lock, self.active_sessions,
                                 self.total_requests, self.successful_requests, self.failed_requests,
                                 self.start_time),
            refresh_per_second=4,
            vertical_overflow="visible",
            auto_refresh=True
        ) as live:
            end_time = time.time() + duration
            session_id = 0
            self.duration = duration
            
            while time.time() < end_time:
                current_time = time.time()
                
                if current_time - self.last_log_time >= 1.0:
                    self.log_status()
                
                if self.active_sessions < self.max_concurrent:
                    with self.lock:
                        self.active_sessions += 1
                    session_id += 1
                    # Run async request in a separate thread to avoid blocking
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, lambda: asyncio.run(self.make_request(session_id)))
                
                if self.should_update_display():
                    live.update(generate_status_table(
                        self.sessions, self.columns, self.lock, self.active_sessions,
                        self.total_requests, self.successful_requests, self.failed_requests,
                        self.start_time
                    ))
                
                await asyncio.sleep(0.1)

def load_dataset_as_questions(dataset_name: str, template: str):
    """
    Load a dataset and format it using the provided template.
    """
    dataset = load_dataset(dataset_name)['train']
    return [template.format(**data) for data in dataset]

async def main(
    model: str = "gpt-3.5-turbo",
    api_url: str = None,
    max_concurrent: int = 5,
    columns: int = 3,
    log_file: str = None,
    output_dir: str = None,
    env: str = None,
    dataset: str = "tatsu-lab/alpaca",
    template: str = "{input}\nQuestion: {instruction}",
    time_limit: int = 10
):
    """
    Main function to run the API throughput monitor.
    """
    global questions
    if env is not None:
        load_dotenv(env)
    
    questions = load_dataset_as_questions(dataset, template)
    
    # Set default values
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    api_url = api_url if api_url is not None else os.environ.get('API_URL')
    api_key = os.environ.get('OPENAI_API_KEY')
    model = model if model is not None else os.environ.get('MODEL', "gpt-3.5-turbo")
    log_file = log_file if log_file is not None else f"{output_dir}/api_monitor.jsonl" if output_dir is not None else "api_monitor.jsonl"
    
    # Initialize monitor
    monitor = APIThroughputMonitor(
        model=model,
        api_url=api_url,
        api_key=api_key,
        max_concurrent=max_concurrent,
        columns=columns,
        log_file=log_file,
        output_dir=output_dir,
    )
    
    # Display configuration
    monitor.logger.info(f"API URL: {api_url}")
    monitor.logger.info(f"Model: {model}")
    monitor.logger.info(f"Log File: {log_file}")
    monitor.logger.info(f"Max Concurrent Requests: {max_concurrent}")
    monitor.logger.info(f"Output Directory: {output_dir}")
    monitor.logger.info(f"Time Limit: {time_limit} seconds")
    
    monitor.logger.info("🚀 Starting API Throughput Monitor...")
    monitor.logger.info("Press Ctrl+C to stop the monitor\n")
    
    try:
        await monitor.run(duration=time_limit)
    except KeyboardInterrupt:
        monitor.logger.info("\n\n👋 Shutting down monitor...")
    finally:
        monitor.logger.info("\n✨ Monitor stopped. Final statistics displayed above.")
        monitor.logger.info(f"Log file saved as: {monitor.log_file}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)