import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from e2e_artifact import atomic_write_text, write_outputs
from html_report import load_report_data


class ArtifactTests(unittest.TestCase):
    def test_only_two_outputs_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            summary = {"run_name": "run", "status": "answering", "config": {"top_k": 5},
                       "dataset": {"selected_sample_count": 2}, "metrics": {"overall": {}}}
            results = [{"id": "a", "answer": "saved", "scoring_status": "pending"}]
            write_outputs(path, results, summary)
            self.assertEqual({p.name for p in path.parent.iterdir()}, {"run.json", "run.html"})
            loaded, metadata = load_report_data(path)
            self.assertEqual(loaded, results)
            self.assertEqual(metadata, summary)
            self.assertIn("未完成评测快照", path.with_suffix(".html").read_text())
            results[0]["scoring_status"] = "completed"
            results[0]["answer_metrics"] = {"correctness_score": 1}
            summary["status"] = "completed"
            write_outputs(path, results, summary)
            self.assertEqual(load_report_data(path)[0][0]["answer_metrics"]["correctness_score"], 1)
            self.assertNotIn("未完成评测快照", path.with_suffix(".html").read_text())

    def test_failed_atomic_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            atomic_write_text(path, "previous")
            with patch.object(Path, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, "new")
            self.assertEqual(path.read_text(), "previous")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_legacy_jsonl_report_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            path.write_text('{"id":"a"}\n')
            path.with_name("old_summary.json").write_text(json.dumps({"run_name": "old"}))
            self.assertEqual(load_report_data(path), ([{"id": "a"}], {"run_name": "old"}))


if __name__ == "__main__":
    unittest.main()
