"""Standard-library tests; no model, database, Docker or network required."""

import contextlib
import io
import logging
import os
from pathlib import Path
import signal
import selectors
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation_logging import BACKUP_COUNT, MAX_BYTES, EvaluationLog, run_logged_script


MODULE_DIRECTORY = Path(__file__).resolve().parent


class EvaluationLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.log = self.root / "logs" / "evaluation.log"

    def script(self, body):
        path = self.root / "evaluate.py"
        path.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(MODULE_DIRECTORY)!r})\n"
            "from evaluation_logging import is_logging_child, run_logged_script\n"
            "if not is_logging_child(__file__):\n"
            f"    raise SystemExit(run_logged_script(__file__, log_path={str(self.log)!r}))\n"
            + body, encoding="utf-8",
        )
        return path

    def run_script(self, body):
        return subprocess.run([sys.executable, str(self.script(body))], capture_output=True, timeout=15)

    def test_creates_and_appends_with_distinct_run_markers(self):
        self.assertFalse(self.log.exists())
        first = self.run_script("print('first-run')\n")
        old = self.log.read_bytes()
        second = self.run_script("print('second-run')\n")
        data = self.log.read_bytes()
        self.assertEqual((first.returncode, second.returncode), (0, 0))
        self.assertTrue(data.startswith(old))
        self.assertEqual(data.count(b" START "), 2)
        self.assertEqual(data.count(b" END "), 2)
        self.assertEqual(data.count(b"status=SUCCESS"), 2)
        starts = [line for line in data.splitlines() if b" START " in line]
        self.assertNotEqual(starts[0].split()[2], starts[1].split()[2])
        for field in (b"started_at=", b"ended_at=", b"duration_seconds=", b"exit_code=0"):
            self.assertIn(field, data)

    def test_all_output_sources_are_captured_once(self):
        result = self.run_script(
            "import logging, os, subprocess\n"
            "print('导入阶段输出', flush=True)\n"
            "sys.stdout.write('partial'); sys.stdout.flush()\n"
            "sys.stdout.write('-line\\n')\n"
            "sys.stderr.write('stderr-token\\n')\n"
            "logging.basicConfig(level=logging.INFO, format='%(message)s')\n"
            "logging.warning('root-token')\n"
            "logger = logging.getLogger('independent')\n"
            "logger.propagate = False\n"
            "logger.addHandler(logging.StreamHandler(sys.stderr))\n"
            "logger.warning('independent-token')\n"
            "os.write(1, b'native-token\\n')\n"
            "sys.stdout.buffer.write(b'binary-token\\n'); sys.stdout.flush()\n"
            "subprocess.run([sys.executable, '-c', \"print('subprocess-token')\"], check=True)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = self.log.read_bytes()
        terminal_lines = (result.stdout + result.stderr).splitlines()
        logged_lines = [line for line in data.splitlines() if line and not line.startswith(b"=" * 24)]
        self.assertCountEqual(logged_lines, terminal_lines)
        self.assertEqual(len(terminal_lines), 8)
        for line in terminal_lines:
            self.assertEqual(logged_lines.count(line), 1)
        self.assertIn(b"stderr-token", result.stderr)
        self.assertNotIn(b"stderr-token", result.stdout)

    def test_unhandled_import_error_and_traceback_saved_once(self):
        result = self.run_script("import evaluation_nonexistent_dependency\n")
        self.assertNotEqual(result.returncode, 0)
        data = self.log.read_bytes()
        self.assertIn(result.stderr, data)
        self.assertEqual(data.count(b"Traceback (most recent call last):"), 1)
        self.assertIn(b"ModuleNotFoundError", data)
        self.assertIn(b"status=FAILED", data)

    def test_nonzero_exit_is_failed(self):
        result = self.run_script("raise SystemExit(7)\n")
        self.assertEqual(result.returncode, 7)
        self.assertIn(b"status=FAILED exit_code=7", self.log.read_bytes())

    def test_output_is_live_and_sigterm_is_forwarded(self):
        script = self.script(
            "import time\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n"
        )
        child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=5), "output was buffered until process exit")
                self.assertEqual(os.read(child.stdout.fileno(), 1024), b"ready\n")
            self.assertIsNone(child.poll())
            child.terminate()
            child.communicate(timeout=5)
            self.assertEqual(child.returncode, 143)
            self.assertIn(b"status=FAILED exit_code=143", self.log.read_bytes())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)

    def test_both_real_entrypoints_capture_before_application_imports(self):
        for name in ("run_e2e_eval.py", "run_retrieval_eval.py"):
            with self.subTest(name=name):
                # Copy the real bootstrap, substituting only its module location
                # and log destination. An import failure must already be captured.
                prefix = (MODULE_DIRECTORY / name).read_text().split("SCRIPT_DIRECTORY =", 1)[0]
                prefix = prefix.replace(
                    'if __name__ == "__main__":',
                    f"sys.path.insert(0, {str(MODULE_DIRECTORY)!r})\nif __name__ == '__main__':",
                ).replace("run_logged_script(__file__)", f"run_logged_script(__file__, log_path={str(self.log)!r})")
                script = self.root / name
                script.write_text(prefix + "raise RuntimeError('application-import-failure')\n", encoding="utf-8")
                result = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 1)
                self.assertIn(b"application-import-failure", result.stderr)
        data = self.log.read_bytes()
        self.assertEqual(data.count(b"status=FAILED"), 2)
        self.assertEqual(data.count(b"Traceback (most recent call last):"), 2)

    def test_streams_unchanged_after_failure(self):
        script = self.root / "failure.py"
        script.write_text("raise ValueError('failed-run')\n", encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        previous_signal = signal.getsignal(signal.SIGTERM)
        with patch.object(sys, "argv", [str(script)]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(run_logged_script(str(script), log_path=self.log), 1)
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_signal)
        self.assertIn("ValueError: failed-run", err.getvalue())

    def test_rotation_retains_three_backups_and_discards_oldest(self):
        self.assertEqual(MAX_BYTES, 50 * 1024 * 1024)
        self.assertEqual(BACKUP_COUNT, 3)
        for letter in b"ABCDEF":
            # Reopen on every write to exercise restart + append + rotation.
            sink = EvaluationLog(self.log, max_bytes=64)
            try:
                sink.write(bytes([letter]) * 64)
            finally:
                sink.close()
        self.assertEqual(self.log.read_bytes(), b"F" * 64)
        for number, letter in enumerate(b"EDC", start=1):
            self.assertEqual(Path(str(self.log) + f".{number}").read_bytes(), bytes([letter]) * 64)
        self.assertFalse(Path(str(self.log) + ".4").exists())

    def test_large_single_write_stays_bounded(self):
        sink = EvaluationLog(self.log, max_bytes=64)
        try:
            sink.write(b"x" * 1000)
        finally:
            sink.close()
        files = [self.log, *self.log.parent.glob("evaluation.log.[123]")]
        self.assertEqual(len(files), 4)
        self.assertTrue(all(path.stat().st_size <= 64 for path in files))
        self.assertLessEqual(sum(path.stat().st_size for path in files), 256)

    def test_no_global_stream_handler_or_signal_changes(self):
        script = self.root / "plain.py"
        script.write_text("print('hello')\n", encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        handlers = list(logging.getLogger().handlers)
        original_signal = signal.getsignal(signal.SIGTERM)
        with patch.object(sys, "argv", [str(script)]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            for _ in range(2):
                self.assertEqual(run_logged_script(str(script), log_path=self.log), 0)
                self.assertIs(sys.stdout, out)
                self.assertIs(sys.stderr, err)
            print("outside-capture")
        self.assertEqual(logging.getLogger().handlers, handlers)
        self.assertEqual(signal.getsignal(signal.SIGTERM), original_signal)
        self.assertEqual(self.log.read_bytes().count(b"hello\n"), 2)
        self.assertNotIn(b"outside-capture", self.log.read_bytes())
        self.assertEqual(out.getvalue(), "hello\nhello\noutside-capture\n")

    def test_multiple_processes_can_rotate_one_log(self):
        writer = self.root / "writer.py"
        writer.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(MODULE_DIRECTORY)!r})\n"
            "from evaluation_logging import EvaluationLog\n"
            f"sink = EvaluationLog({str(self.log)!r}, max_bytes=256)\n"
            "try:\n"
            "    for i in range(60): sink.write((sys.argv[1] * 31 + '\\n').encode())\n"
            "finally: sink.close()\n", encoding="utf-8",
        )
        children = [subprocess.Popen([sys.executable, str(writer), letter], stderr=subprocess.PIPE)
                    for letter in "ABC"]
        for child in children:
            _, err = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 0, err)
        files = [self.log, *self.log.parent.glob("evaluation.log.[123]")]
        self.assertEqual(len(files), 4)
        for path in files:
            self.assertLessEqual(path.stat().st_size, 256)
            for line in path.read_bytes().splitlines():
                self.assertIn(line, [letter.encode() * 31 for letter in "ABC"])

    def test_threaded_output_is_not_lost(self):
        result = self.run_script(
            "from concurrent.futures import ThreadPoolExecutor\n"
            "import os\n"
            "with ThreadPoolExecutor(max_workers=8) as pool:\n"
            "    list(pool.map(lambda n: os.write(1, f'thread-{n}\\n'.encode()), range(200)))\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = self.log.read_bytes()
        for n in range(200):
            self.assertEqual(data.splitlines().count(f"thread-{n}".encode()), 1)


if __name__ == "__main__":
    unittest.main()
