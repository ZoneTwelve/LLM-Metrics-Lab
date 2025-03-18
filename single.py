import requests
import time
import json
from datetime import datetime
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
import os
from dotenv import load_dotenv
import logging
import socket

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SingleRequestMonitor")

class SingleRequestMonitor:
    def __init__(self, model: str, api_url: str, api_key: str):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.console = Console()
        self.response_text = ""
        self.start_time = None
        self.first_token_time = None
        self.chunks_received = 0
        self.status = "Idle"

    def generate_status_table(self):
        table = Table(
            title="API Request Monitor",
            box=box.ROUNDED,
            title_style="bold magenta",
            header_style="bold cyan",
            show_lines=True
        )

        table.add_column("Status", width=15)
        table.add_column("Details", width=80)

        elapsed_time = time.time() - self.start_time if self.start_time else 0
        status_style = {
            "Idle": "white",
            "Processing": "blue",
            "Completed": "green",
            "Failed": "red"
        }.get(self.status, "white")

        table.add_row(
            f"[{status_style}]Status[/{status_style}]",
            f"[{status_style}]{self.status}[/{status_style}]"
        )
        table.add_row("Elapsed Time", f"{elapsed_time:.2f}s")
        table.add_row("Chunks Received", str(self.chunks_received))
        first_token_latency = (self.first_token_time - self.start_time) if self.first_token_time else 0
        table.add_row("First Token Latency", f"{first_token_latency:.2f}s" if self.first_token_time else "N/A")
        table.add_row("Response", self.response_text or "Waiting for response...")

        return table

    def process_stream_line(self, line):
        try:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]
            if line.strip() == '[DONE]':
                return None
            data = json.loads(line)
            if 'choices' in data and len(data['choices']) > 0:
                if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                    return data['choices'][0]['delta']['content']
            return None
        except Exception as e:
            logger.error(f"Error processing stream line: {e}")
            return None

    def make_request(self, prompt: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            self.start_time = time.time()
            self.status = "Processing"
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                verify=False,
                timeout=180
            )

            for line in response.iter_lines():
                if line:
                    content = self.process_stream_line(line)
                    if content is None:
                        break
                    if not self.first_token_time:
                        self.first_token_time = time.time()
                    self.chunks_received += 1
                    self.response_text += content
                    yield

            self.status = "Completed"

        except Exception as e:
            self.status = "Failed"
            logger.error(f"Request failed: {str(e)}")
            self.response_text = f"Error: {str(e)}"

    def run(self, prompt: str):
        with Live(
            self.generate_status_table(),
            refresh_per_second=10,
            console=self.console,
            vertical_overflow="visible"
        ) as live:
            for _ in self.make_request(prompt):
                live.update(self.generate_status_table())
            live.update(self.generate_status_table())

def load_prompt_from_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file {file_path}: {str(e)}")
        raise

def connect_to_cc_server(host: str, port: int):
    """Connect to the C&C server and wait for the 'start' signal."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        logger.info(f"Connected to C&C Server at {host}:{port}")
        logger.info("Waiting for 'start' signal from C&C server...")

        while True:
            data = client_socket.recv(1024).decode('utf-8')
            if data == "start":
                logger.info("Received 'start' signal from C&C server")
                break
            elif data:
                logger.info(f"Received unexpected data: {data}")
            time.sleep(0.1)  # Small delay to prevent busy-waiting
    except Exception as e:
        logger.error(f"Error connecting to C&C server: {e}")
        raise
    finally:
        client_socket.close()

def main(
    model: str = "gpt-3.5-turbo",
    api_url: str = None,
    prompt: str = None,
    prompt_file: str = None,
    cc_host: str = "zonetwelve-local",
    cc_port: int = 9999,
    delay: int = 10
):
    api_url = api_url if api_url is not None else os.environ.get('API_URL')
    api_key = os.environ.get('API_KEY')

    if not api_url or not api_key:
        logger.error("API_URL and API_KEY must be set in environment variables or provided as arguments")
        return

    # Handle prompt input
    if prompt and prompt_file:
        logger.error("Please provide either --prompt or --prompt-file, not both")
        return
    elif prompt_file:
        try:
            prompt_text = load_prompt_from_file(prompt_file)
        except Exception:
            return
    elif prompt:
        prompt_text = prompt
    else:
        prompt_text = "Tell me a short story about a curious cat."

    logger.info(f"Model: {model}")
    logger.info(f"API URL: {api_url}")
    logger.info(f"C&C Server: {cc_host}:{cc_port}")
    logger.info(f"Prompt: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")

    # Wait for C&C server start signal
    try:
        connect_to_cc_server(cc_host, cc_port)
    except Exception:
        return
    logger.info(f"Wait {delay} seconds before starting the request...")
    time.sleep(delay)
    logger.info(f"Starting single request monitor...")

    monitor = SingleRequestMonitor(
        model=model,
        api_url=api_url,
        api_key=api_key
    )
    
    try:
        monitor.run(prompt_text)
        logger.info("Request completed")
    except KeyboardInterrupt:
        logger.info("Stopped by user")

if __name__ == "__main__":
    import fire
    fire.Fire(main)