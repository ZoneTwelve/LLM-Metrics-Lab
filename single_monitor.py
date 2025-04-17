import asyncio
import time
import json
from datetime import datetime
from rich.live import Live
from rich.console import Console
import os
from dotenv import load_dotenv
import socket

from utils.utils import OpenAIPayload, OpenAIAPIRequest, openai_compatible_request, process_stream_chunk
from utils.monitor import setup_logging

load_dotenv()

class SingleRequestMonitor:
    def __init__(self, model: str, api_url: str, api_key: str):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.console = Console()
        self.logger = setup_logging(
            console_log_level=os.getenv("CLL", "INFO"),
            file_log_level=os.getenv("FLL", ""),
            log_file="single_monitor.log"
        )
        self.reset()

    def reset(self):
        """Reset timing and response variables for a new request."""
        self.response_text = ""
        self.start_time = None
        self.first_token_time = None
        self.chunks_received = 0
        self.status = "Idle"

    def generate_status_table(self):
        """
        Generate a status table for the current request.
        """
        from rich.table import Table
        from rich import box

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

    async def make_request(self, prompt: str):
        """
        Make an asynchronous streaming request using utils.py's openai_compatible_request.
        """
        payload = OpenAIPayload(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        request = OpenAIAPIRequest(
            api_url=f"{self.api_url}/chat/completions",
            api_key=self.api_key,
            payload=payload
        )

        try:
            self.start_time = time.time()
            self.status = "Processing"

            async for chunk in await openai_compatible_request(request):
                processed_chunk = process_stream_chunk(chunk)
                content = processed_chunk.get("content", "")
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
            self.logger.error(f"Request failed: {str(e)}")
            self.response_text = f"Error: {str(e)}"

    async def run(self, prompt: str):
        """
        Run the monitor for a single request, updating the status table live.
        """
        with Live(
            self.generate_status_table(),
            refresh_per_second=10,
            console=self.console,
            vertical_overflow="visible"
        ) as live:
            async for _ in self.make_request(prompt):
                live.update(self.generate_status_table())
            live.update(self.generate_status_table())

def load_prompt_from_file(file_path: str) -> str:
    """
    Load a prompt from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file {file_path}: {str(e)}")
        raise

def connect_to_cc_server(host: str, port: int, logger):
    """
    Connect to the C&C server and wait for the 'start' signal.
    """
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
        logger.error(f"Error connecting to C&C server: Quantitative {e}")
        raise
    finally:
        client_socket.close()

async def main(
    model: str = "gpt-3.5-turbo",
    api_url: str = None,
    prompt: str = None,
    prompt_file: str = None,
    cc_host: str = "zonetwelve-local",
    cc_port: int = 9999,
    delay: int = 10,
    loop: int = 1,
    interval: int = 0
):
    """
    Main function to run the single request monitor.
    """
    logger = setup_logging(
        console_log_level=os.getenv("CLL", "INFO"),
        file_log_level=os.getenv("FLL", ""),
        log_file="single_monitor.log"
    )

    api_url = api_url if api_url is not None else os.environ.get('API_URL')
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_url or not api_key:
        logger.error("API_URL and OPENAI_API_KEY must be set in environment variables or provided as arguments")
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
        prompt_text = "Tell me a story about a curious cat."

    logger.info(f"Model: {model}")
    logger.info(f"API URL: {api_url}")
    # logger.info(f"C&C Server: {cc_host}:{cc_port}")
    logger.info(f"Initial Delay: {delay}s")
    logger.info(f"Loop Count: {loop if loop != -1 else 'Infinite'}")
    logger.info(f"Interval between loops: {interval}s")
    logger.info(f"Prompt: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")

    # Wait for C&C server start signal
    # try:
    #     connect_to_cc_server(cc_host, cc_port, logger)
    # except Exception:
    #     return

    logger.info(f"Waiting {delay} seconds before starting the request(s)...")
    time.sleep(delay)

    monitor = SingleRequestMonitor(
        model=model,
        api_url=api_url,
        api_key=api_key
    )

    try:
        loop_count = 0
        while loop == -1 or loop_count < loop:
            logger.info(f"Starting request {loop_count + 1}{' (Infinite mode)' if loop == -1 else ''}")
            monitor.reset()  # Reset the monitor before each new request
            await monitor.run(prompt_text)
            loop_count += 1

            if (loop == -1 or loop_count < loop) and interval > 0:
                logger.info(f"Waiting {interval} seconds before next request...")
                time.sleep(interval)

        logger.info("All requests completed")
    except KeyboardInterrupt:
        logger.info("Stopped by user")

if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv
    load_dotenv()
    fire.Fire(main)