"""Tests for utils/logger.py — ExperimentLogger."""
import json
import os
import pytest

from utils.logger import ExperimentLogger


class TestExperimentLogger:
    def test_basic_log(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ExperimentLogger(log_dir=str(log_dir), run_name="test_run")
        logger.log(epoch=1, loss=0.5, acc=0.8)
        logger.close()

        fp = log_dir / "test_run.jsonl"
        assert fp.exists()
        lines = fp.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["epoch"] == 1
        assert record["loss"] == 0.5
        assert record["acc"] == 0.8
        assert "timestamp" in record

    def test_multiple_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ExperimentLogger(log_dir=str(log_dir), run_name="multi")
        for i in range(5):
            logger.log(step=i, value=i * 10)
        logger.close()

        fp = log_dir / "multi.jsonl"
        lines = fp.read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["step"] == i
            assert record["value"] == i * 10

    def test_context_manager(self, tmp_path):
        log_dir = tmp_path / "logs"
        with ExperimentLogger(log_dir=str(log_dir), run_name="ctx") as logger:
            logger.log(msg="hello")

        fp = log_dir / "ctx.jsonl"
        assert fp.exists()
        record = json.loads(fp.read_text().strip())
        assert record["msg"] == "hello"

    def test_auto_run_name(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ExperimentLogger(log_dir=str(log_dir))
        logger.log(test=True)
        logger.close()
        # Should create a file with timestamp-based name
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1

    def test_close_idempotent(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = ExperimentLogger(log_dir=str(log_dir), run_name="idem")
        logger.log(a=1)
        logger.close()
        logger.close()  # should not raise

    def test_flush_on_log(self, tmp_path):
        """After log(), file should be readable (flushed)."""
        log_dir = tmp_path / "logs"
        logger = ExperimentLogger(log_dir=str(log_dir), run_name="flush")
        logger.log(x=42)
        # Read without closing — should be available due to flush
        fp = log_dir / "flush.jsonl"
        content = fp.read_text().strip()
        assert "42" in content
        logger.close()
