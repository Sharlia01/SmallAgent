"""无需数据库或模型调用的端到端汇总契约测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from evaluation.end_to_end import build_e2e_summary


def record(identifier="a", mode="answer"):
    return {
        "id": identifier, "question_type": "fact", "answerable": mode == "answer",
        "response_mode": mode, "outcome": "success" if mode == "answer" else "insufficient_evidence",
        "error": None, "latency_ms": {"total": 100}, "retrieval": {"calls": []},
        "answer_metrics": {"correctness_score": 1, "completeness_score": 1} if mode == "answer" else {},
        "faithfulness_metrics": {"faithfulness_score": 1} if mode == "answer" else {},
        "citation_metrics": {"citation_recall": 1} if mode == "answer" else {},
        "refusal_metrics": {"expected_behavior": mode, "actual_behavior": mode,
                            "correct": True, "unsafe_answer": False, "over_refusal": False},
        "sufficiency_metrics": {"expected_sufficient": mode == "answer", "actual_sufficient": mode == "answer",
                                "correct": True, "unsafe_accept": False, "false_rejection": False},
    }


class SummaryTests(unittest.TestCase):
    def test_main_and_diagnostics_are_separate(self):
        summary = build_e2e_summary([record(), record("b", "refuse")])
        main = summary["overall"]
        self.assertEqual(summary["summary_schema_version"], "4.0")
        for key in ("retrieval_metrics", "post_gate_metrics", "successful_query_count", "query_metrics",
                    "faithfulness_micro", "citation_precision", "non_perfect_sample_ids"):
            self.assertNotIn(key, main)
        self.assertIn("retrieval_call_metrics", main["diagnostics"])
        self.assertNotIn("route_agreement", main["diagnostics"]["query_metrics"])
        self.assertEqual(main["scoring"]["answer_correctness"]["applicable_count"], 1)
        self.assertEqual(main["answer_correctness"], 1)
        self.assertIn("diagnostics", summary["by_question_type"]["fact"])

    def test_internal_failure_is_counted_even_without_top_level_error(self):
        row = record(mode="refuse")
        row["outcome"] = "execution_error"
        main = build_e2e_summary([row])["overall"]
        self.assertEqual(main["execution_failure_count"], 1)
        self.assertIn("execution_failure", main["sample_issues"][0]["metrics"])

    def test_goal_diagnostics_and_identity_failure_survive_summary(self):
        row = record(mode="refuse")
        row["outcome"] = "execution_error"
        row["retrieval"].update({
            "query_understanding": {"goals": [{"id": "g1"}, {"id": "g2"}]},
            "evidence_sufficiency": {"verification_status": "unavailable", "goal_coverage": [],
                                     "failure_kind": "EvidenceIdentityConflict"},
        })
        main = build_e2e_summary([row])["overall"]
        goals = main["diagnostics"]["query_metrics"]
        self.assertEqual(goals["planned_goal_count"], 2)
        self.assertEqual(goals["evaluated_goal_count"], 0)
        self.assertEqual(goals["unevaluated_goal_count"], 2)
        self.assertIsNone(goals["supported_goal_rate"])
        self.assertEqual(goals["goal_metric_scope"], "planned_goals_only")
        self.assertEqual(main["execution_failure_count"], 1)

    def test_judge_failure_does_not_disappear_from_denominator(self):
        row = record()
        row["answer_metrics"] = {"judge_error": "timeout"}
        main = build_e2e_summary([row])["overall"]
        self.assertIsNone(main["answer_correctness"])
        self.assertEqual(main["scoring"]["answer_correctness"], {
            "applicable_count": 1, "evaluated_count": 0,
            "judge_failure_count": 1, "not_applicable_count": 0})
        self.assertIn("answer_metrics.judge_failure", main["sample_issues"][0]["metrics"])

    def test_empty_run_has_no_fabricated_scores(self):
        main = build_e2e_summary([])["overall"]
        self.assertIsNone(main["answer_correctness"])
        self.assertEqual(main["execution_failure_count"], 0)

    def test_no_factual_claims_are_not_judge_failures(self):
        row = record()
        row["faithfulness_metrics"] = {"factual_claim_count": 0, "faithfulness_score": None}
        counts = build_e2e_summary([row])["overall"]["scoring"]["faithfulness"]
        self.assertEqual(counts["applicable_count"], 0)
        self.assertEqual(counts["not_applicable_count"], 1)
        self.assertEqual(counts["judge_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
