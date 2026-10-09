"""运行方式：python -m unittest discover -s evals -p test_html_report.py。"""

import unittest

from html_report import render_html_report


class HtmlReportTests(unittest.TestCase):
    def test_untrusted_text_is_escaped_and_raw_trace_is_omitted(self):
        report = render_html_report([{
            "question": '<script>alert("question")</script>',
            "answer": '<img src=x onerror="alert(1)">',
            "reference_answer": "A & B",
            "error": "<unsafe-error>",
            "retrieval": {"chunks": [{"content": "PRIVATE_RAW_TRACE"}]},
            "thinking": "PRIVATE_THINKING",
        }], {"run_name": "<untrusted-title>"})
        self.assertIn("&lt;script&gt;", report)
        self.assertNotIn('<img src=x', report)
        self.assertIn("A &amp; B", report)
        self.assertIn("&lt;unsafe-error&gt;", report)
        self.assertNotIn("PRIVATE_RAW_TRACE", report)
        self.assertNotIn("PRIVATE_THINKING", report)

    def test_judge_failure_is_not_a_zero_or_a_success(self):
        report = render_html_report([{
            "faithfulness_metrics": {
                "faithfulness_score": None,
                "judge_error": "ValueError: invalid evidence",
            },
            "answer_metrics": {"correctness_score": 0, "completeness_score": 1},
        }], {})
        self.assertIn('忠实度 <b>未评分</b>', report)
        self.assertIn('正确性 <b>0.0%</b>', report)
        self.assertIn('完整性 <b>100.0%</b>', report)
        self.assertIn('data-attention="true"', report)
        self.assertIn("忠实度评分失败：ValueError: invalid evidence", report)

    def test_refusal_and_citation_failures_are_visible(self):
        report = render_html_report([{
            "refusal_metrics": {"correct": False, "expected_behavior": "refuse", "actual_behavior": "answer"},
            "citation_metrics": {
                "citation_recall": 0,
                "claims": [{"requires_citation": True, "valid_citation_ids": [], "text": "利润翻倍", "judge_reason": "缺少引用"}],
            },
        }], {})
        self.assertIn("应为 refuse，实际为 answer", report)
        self.assertIn("利润翻倍；缺少引用", report)

    def test_disabled_scores_are_unavailable_without_failure(self):
        report = render_html_report([{"answer_metrics": {"judge_source": "disabled"}}], {})
        self.assertIn('data-attention="false"', report)
        self.assertIn("部分指标未评分", report)
        self.assertNotIn("答案评分失败", report)


if __name__ == "__main__":
    unittest.main()
