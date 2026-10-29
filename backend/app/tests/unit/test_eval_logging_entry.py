"""Exercise the actual CLI bootstraps with application dependencies installed."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("entry", ["run_e2e_eval.py", "run_retrieval_eval.py"])
def test_validation_cli_automatically_captures_and_appends(tmp_path, entry):
    app = Path(__file__).resolve().parents[2]
    source = app.parent / "evals"
    scripts = tmp_path / "evals"
    scripts.mkdir()
    for name in ("evaluation_logging.py", "run_e2e_eval.py", "run_retrieval_eval.py", "html_report.py"):
        shutil.copyfile(source / name, scripts / name)
    env = {**os.environ, "PYTHONPATH": str(app) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    command = [sys.executable, str(scripts / entry), "--validate-only", "--limit", "1",
               "--dataset", str(source / "data" / "power_reports_eval_v2_dev.jsonl")]
    log = tmp_path / "logs" / "evaluation.log"
    previous = b""
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, env=env, timeout=45)
        assert result.returncode == 0, result.stderr.decode()
        assert json.loads(result.stdout)["selected_samples"] == 1
        content = log.read_bytes()
        assert content.startswith(previous)
        assert result.stdout in content[len(previous):]
        previous = content
    assert previous.count(b" START ") == 2
    assert previous.count(b"status=SUCCESS exit_code=0") == 2
    # Startup argument failures are captured too and preserve argparse's status.
    failure = subprocess.run(command + ["--unknown-option"], capture_output=True, env=env, timeout=45)
    assert failure.returncode == 2
    content = log.read_bytes()
    assert b"unrecognized arguments: --unknown-option" in content
    assert b"status=FAILED exit_code=2" in content
