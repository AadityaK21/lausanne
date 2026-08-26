"""
Tests for the conversation/context-management layer.

None of these load a model — `Conversation` is deliberately pure Python so the
trimming rules can be tested exhaustively and instantly. The trimming logic is
where multi-turn chatbots actually break, so it gets the coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a2n.chatbot import (  # noqa: E402
    Conversation, EchoModel, Message, estimate_tokens,
)


# --------------------------------------------------------------- basics
def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("hi") < estimate_tokens("hi " * 100)
    assert estimate_tokens("") >= 1          # never zero, avoids div-by-zero downstream


def test_roles_are_recorded_in_order():
    c = Conversation()
    c.add_user("one")
    c.add_assistant("two")
    c.add_user("three")
    assert [m.role for m in c.messages] == ["user", "assistant", "user"]


def test_reset_clears_history_but_keeps_system_prompt():
    c = Conversation(system_prompt="STAY")
    c.add_user("x")
    c.reset()
    assert c.messages == []
    assert c.build_prompt_messages()[0] == {"role": "system", "content": "STAY"}


# ------------------------------------------------------ prompt assembly
def test_system_prompt_is_always_first_and_present():
    c = Conversation(system_prompt="SYSTEM")
    for i in range(20):
        c.add_user(f"u{i}")
        c.add_assistant(f"a{i}")
    out = c.build_prompt_messages()
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "SYSTEM"
    assert sum(1 for m in out if m["role"] == "system") == 1


def test_short_conversation_is_not_trimmed():
    c = Conversation(max_context_tokens=4096)
    c.add_user("hello")
    c.add_assistant("hi")
    assert not c.was_trimmed()
    assert len(c.build_prompt_messages()) == 3      # system + 2


def test_long_conversation_is_trimmed_to_budget():
    c = Conversation(max_context_tokens=400, reserve_for_reply=100)
    for i in range(60):
        c.add_user("u" * 200)
        c.add_assistant("a" * 200)
    assert c.was_trimmed()
    out = c.build_prompt_messages()
    total = sum(estimate_tokens(m["content"]) for m in out)
    # Allow one block of overshoot: the newest block is always kept even if it
    # alone exceeds the budget, otherwise the model would receive no user turn.
    assert total <= c.max_context_tokens


def test_trimming_keeps_the_most_recent_turns():
    c = Conversation(max_context_tokens=300, reserve_for_reply=50)
    for i in range(40):
        c.add_user(f"question number {i} " + "x" * 100)
        c.add_assistant(f"answer number {i} " + "y" * 100)
    out = c.build_prompt_messages()
    assert "question number 39" in out[-2]["content"]
    assert "answer number 39" in out[-1]["content"]
    assert not any("question number 0 " in m["content"] for m in out)


def test_trimming_never_orphans_an_assistant_reply():
    """
    An assistant message whose user prompt was evicted leaves the model reading
    an answer to a question it cannot see. Eviction happens in whole pairs, so
    the kept history must never start with an assistant turn.
    """
    for budget in (120, 200, 350, 500, 900):
        c = Conversation(max_context_tokens=budget, reserve_for_reply=40)
        for i in range(30):
            c.add_user(f"u{i} " + "x" * 80)
            c.add_assistant(f"a{i} " + "y" * 80)
        history = c.build_prompt_messages()[1:]     # drop the system message
        assert history, f"budget {budget}: everything was trimmed away"
        assert history[0]["role"] == "user", f"budget {budget}: orphaned assistant reply"


def test_newest_turn_survives_even_when_it_alone_exceeds_budget():
    c = Conversation(max_context_tokens=100, reserve_for_reply=20)
    c.add_user("tiny")
    c.add_assistant("tiny")
    c.add_user("z" * 5000)
    out = c.build_prompt_messages()
    assert out[-1]["content"].startswith("z")


def test_a_huge_system_prompt_does_not_produce_a_negative_budget_crash():
    c = Conversation(system_prompt="s" * 20000, max_context_tokens=500)
    c.add_user("hello")
    out = c.build_prompt_messages()          # must not raise
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "hello"


# ---------------------------------------------------------- persistence
def test_save_and_load_round_trip(tmp_path):
    c = Conversation(system_prompt="ROUND TRIP", max_context_tokens=999)
    c.add_user("q")
    c.add_assistant("a")
    path = tmp_path / "conv.json"
    c.save(path)

    loaded = Conversation.load(path)
    assert loaded.system_prompt == "ROUND TRIP"
    assert loaded.max_context_tokens == 999
    assert [(m.role, m.content) for m in loaded.messages] == [("user", "q"), ("assistant", "a")]


# ----------------------------------------------------------- echo model
def test_echo_model_sees_the_latest_user_message():
    c = Conversation()
    c.add_user("first")
    c.add_assistant("ok")
    c.add_user("second")
    reply = EchoModel().reply(c.build_prompt_messages())
    assert "second" in reply
    assert "turn 2" in reply


def test_full_loop_stays_consistent_over_many_turns():
    """End-to-end: history grows, trimming kicks in, nothing raises."""
    c = Conversation(max_context_tokens=400, reserve_for_reply=80)
    model = EchoModel()
    for i in range(50):
        c.add_user(f"message {i} " + "w" * 60)
        c.add_assistant(model.reply(c.build_prompt_messages()))
    assert len(c.messages) == 100
    assert c.was_trimmed()
    history = c.build_prompt_messages()[1:]
    assert history[0]["role"] == "user"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-header"]))
