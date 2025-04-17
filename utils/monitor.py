# utils/monitor.py
import logging
import os
from datetime import datetime
import json
import math
from rich.table import Table
from rich.console import Console
from rich import box
import urllib3
import time

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# TODO: move logging setup to a separate module

def setup_logging(console_log_level: str = "INFO", file_log_level: str = "", log_file: str = "metrics.log"):
    """
    Configure logging for console and file output.
    """
    logger = logging.getLogger("Metrics")
    logger.setLevel(getattr(logging, console_log_level.upper(), logging.DEBUG))
    
    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []
    
    if console_log_level:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level.upper())
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file_log_level and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class FileHandler:
    """
    Context manager for handling file operations, with support for virtual (no-op) mode.
    """
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

def generate_status_table(sessions: dict, columns: int, lock, active_sessions: int, total_requests: int,
                         successful_requests: int, failed_requests: int, start_time: float) -> Table:
    """
    Generate a rich table for displaying session statuses.
    """
    console = Console()
    table = Table(
        title="API Throughput Monitor",
        box=box.ROUNDED,
        title_style="bold magenta",
        header_style="bold cyan",
    )
    
    for i in range(columns):
        table.add_column(f"Session Group {i+1}", justify="left")
    
    with lock:
        sorted_sessions = sorted(sessions.items(), key=lambda x: int(x[0]))
        num_sessions = len(sorted_sessions)
        num_rows = math.ceil(num_sessions / columns)
        
        for row_idx in range(num_rows):
            row_data = []
            for col_idx in range(columns):
                session_idx = row_idx * columns + col_idx
                if session_idx < len(sorted_sessions):
                    session_id, info = sorted_sessions[session_idx]
                    status_style = {
                        "Starting": "yellow",
                        "Processing": "blue",
                        "Completed": "green",
                        "Failed": "red"
                    }.get(info["status"], "white")
                    row_data.append(
                        f"{session_id:3d} | "
                        f"[{status_style}]{info['status']:10}[/{status_style}] | "
                        f"Time: {info['response_time'] or '-':8} | "
                        f"Chars: {info['total_chars']:5} | "
                        f"Chunks: {info['chunks_received']:3}"
                    )
                else:
                    row_data.append("")
            table.add_row(*row_data)
        
        elapsed_time = time.time() - start_time
        total_chars = sum(s["total_chars"] for s in sessions.values())
        total_chunks = sum(s["chunks_received"] for s in sessions.values())
        chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0
        
        table.add_section()
        stats_summary = (
            f"[bold cyan]Summary Stats:[/bold cyan]\n"
            f"Time: {elapsed_time:.1f}s \n"
            f"Active: {active_sessions} | "
            f"Total: {total_requests} | "
            f"Success: {successful_requests} | "
            f"Failed: {failed_requests}\n"
            f"Chars/s: {chars_per_sec:.1f} | "
            f"Total Chars: {total_chars} | "
            f"Total Chunks: {total_chunks}"
        )
        table.add_row(stats_summary)
    
    return table
