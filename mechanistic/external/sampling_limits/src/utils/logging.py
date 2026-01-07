import json
import os
from datetime import datetime
import threading

class ExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, "log.jsonl")
        self.lock = threading.Lock()
        
    def log(self, data: dict):
        data['timestamp'] = datetime.now().isoformat()
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
