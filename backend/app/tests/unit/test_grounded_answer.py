"""独立问答入口不再包含自己的理解、检索和回答流程。"""
from service.core.answering import grounded_answer, turn


def test_grounded_entry_forwards_options_without_reinterpreting_question(monkeypatch):
    calls = []
    monkeypatch.setattr(turn, "execute_agent_turn", lambda **kwargs: calls.append(kwargs) or "done")
    history = [{"role": "user", "content": "甲公司"}]
    assert grounded_answer.execute_grounded_turn(index_names="test", question="它的收入？",
        conversation_history=history, retrieval_options={"page_size": 7}) == "done"
    assert calls[0]["question"] == "它的收入？"
    assert calls[0]["conversation_history"] is history
    assert calls[0]["retrieval_options"] == {"page_size": 7}
