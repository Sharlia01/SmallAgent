import pytest

from service.core.deepdoc.vision.recognizer import Recognizer


@pytest.mark.unit
def test_layouts_cleanup_keeps_disjoint_child_tables():
    layouts = [
        {
            "name": "oversized-parent",
            "type": "table",
            "score": 0.5100335,
            "x0": 53.85,
            "x1": 541.65,
            "top": 114.80,
            "bottom": 661.03,
        },
        {
            "name": "balance-sheet",
            "type": "table",
            "score": 0.50101745,
            "x0": 55.43,
            "x1": 293.18,
            "top": 119.68,
            "bottom": 429.46,
        },
        {
            "name": "income-statement",
            "type": "table",
            "score": 0.5221346,
            "x0": 302.18,
            "x1": 539.52,
            "top": 120.03,
            "bottom": 359.37,
        },
        {
            "name": "cash-flow",
            "type": "table",
            "score": 0.2544089,
            "x0": 302.00,
            "x1": 540.00,
            "top": 361.00,
            "bottom": 661.00,
        },
        {
            "name": "key-metrics",
            "type": "table",
            "score": 0.4600538,
            "x0": 55.00,
            "x1": 293.00,
            "top": 444.00,
            "bottom": 659.00,
        },
    ]

    cleaned = Recognizer.layouts_cleanup([], layouts)

    assert [layout["name"] for layout in cleaned] == [
        "balance-sheet",
        "income-statement",
        "cash-flow",
        "key-metrics",
    ]
