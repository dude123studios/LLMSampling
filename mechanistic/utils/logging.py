import json
import os
import time
from typing import Dict, Any, List
from threading import Lock

class ExperimentLogger:
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Create a dedicated folder for this run
        self.run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.run_dir, "results.jsonl")
        self.lock = Lock()
        
        print(f"Logging run to folder: {self.run_dir}")
        print(f"Main log file: {self.log_file}")

    def log(self, data: Dict[str, Any]):
        """Thread-safe logging of a single dictionary record."""
        entry = {
            "timestamp": time.time(),
            **data
        }
        json_str = json.dumps(entry)
        
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json_str + "\n")

    def log_batch(self, data_list: List[Dict[str, Any]]):
        """Log a batch of records."""
        with self.lock:
            with open(self.log_file, "a") as f:
                for data in data_list:
                    entry = {
                        "timestamp": time.time(),
                        **data
                    }
                    f.write(json.dumps(entry) + "\n")

def load_logs(log_file: str) -> List[Dict[str, Any]]:
    """Helper to read logs back."""
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
