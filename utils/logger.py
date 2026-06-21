"""
Simple experiment logger. Replace with wandb / MLflow as needed.
"""
import json
import time
from pathlib import Path


class ExperimentLogger:
    """Minimal key-value logger that flushes to JSON lines."""

    def __init__(self, log_dir="logs", run_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        self.fp = self.log_dir / f"{run_name}.jsonl"
        self._file = open(self.fp, "a")

    def log(self, **kwargs):
        record = {"timestamp": time.time(), **kwargs}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
