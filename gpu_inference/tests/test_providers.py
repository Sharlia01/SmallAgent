import numpy as np
import pytest

from providers import convert_scores


def test_score_conversion_round_trip():
    logits = np.asarray([-2.0, 0.0, 2.0], dtype=float)

    probabilities = convert_scores(
        logits,
        source_mode="raw_logits",
        target_mode="probability",
    )
    restored = convert_scores(
        probabilities,
        source_mode="probability",
        target_mode="raw_logits",
    )

    np.testing.assert_allclose(restored, logits)


def test_probability_conversion_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="between zero and one"):
        convert_scores(
            np.asarray([1.1]),
            source_mode="probability",
            target_mode="raw_logits",
        )

