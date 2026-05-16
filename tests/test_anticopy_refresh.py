"""Schema v2 rollout payload builder (full trajectory + assistant mask)."""

from __future__ import annotations

import pytest

from affine.src.anticopy.refresh import RolloutRefreshService, _SkipSample


class _FakeTokenizer:
    """Stand-in that mimics the surface the builder exercises:

      - ``apply_chat_template(conv, tokenize=False)`` returns a rendered
        ChatML-style text so the assistant-content regex can fire.
      - ``__call__(text, ..., return_offsets_mapping=True)`` returns an
        object with ``.input_ids`` / ``.offset_mapping`` aligned to
        ``text``. Each token is one character (offsets are 1-byte
        slices) — keeps the mask alignment trivial to reason about.
    """

    def apply_chat_template(self, conv, add_generation_prompt=False, tokenize=False):
        # Render every turn as ``<|im_start|>{role}\n{content}<|im_end|>\n``.
        parts = []
        for msg in conv:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        class _R:
            pass

        r = _R()
        r.input_ids = [ord(c) % 1000 for c in text]
        r.offset_mapping = [(i, i + 1) for i in range(len(text))]
        return r


def _sample(extra):
    return {"extra": extra}


def _build(conv):
    return RolloutRefreshService._build_rollout_payload(
        sample=_sample({"conversation": conv}),
        env="MEMORY",
        task_id=1,
        champion_hotkey="HK",
        champion_revision="rev",
        tokenizer=_FakeTokenizer(),
        tokenizer_sig="sig",
        day="2026-05-15",
    )


def test_schema_v2_shape():
    conv = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    payload = _build(conv)
    assert payload["schema"] == "ceac.rollout/v2"
    assert payload["champion"] == {"hotkey": "HK", "revision": "rev"}
    assert payload["env"] == "MEMORY"
    assert payload["task_id"] == 1
    assert payload["tokenizer_sig"] == "sig"
    assert isinstance(payload["tokens"], list) and payload["tokens"]
    assert len(payload["tokens"]) == len(payload["assistant_mask"])


def test_assistant_mask_marks_assistant_content_only():
    """The assistant_mask should be True for the 'hi' characters and
    False for the system/user content or the chat-template framing."""
    conv = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "AB"},
        {"role": "user", "content": "next"},
    ]
    payload = _build(conv)
    text = "".join(  # mirror tokenizer rendering
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conv
    )
    # Each char is one token; find positions of "AB"
    a_pos = text.find("AB")
    b_pos = a_pos + 1
    assert payload["assistant_mask"][a_pos] is True
    assert payload["assistant_mask"][b_pos] is True
    # System / user content slots should be False
    s_pos = text.find("S", 0)
    u_pos = text.find("U", 0)
    assert payload["assistant_mask"][s_pos] is False
    assert payload["assistant_mask"][u_pos] is False


def test_multi_turn_marks_every_assistant_turn():
    """All assistant turns contribute to the mask, not just the last."""
    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a3"},
    ]
    payload = _build(conv)
    text = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conv
    )
    mask = payload["assistant_mask"]
    for content in ("a1", "a2", "a3"):
        # every char of every assistant content must be flagged
        pos = text.find(content)
        assert mask[pos] is True
        assert mask[pos + 1] is True
    for content in ("q1", "q2", "t1"):
        pos = text.find(content)
        assert mask[pos] is False
        assert mask[pos + 1] is False


def test_skip_when_no_assistant_with_content():
    """All assistant turns empty (only tool_calls) → skip."""
    conv = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "ok"},
    ]
    with pytest.raises(_SkipSample):
        _build(conv)


def test_skip_when_conversation_too_short():
    with pytest.raises(_SkipSample):
        _build([{"role": "user", "content": "alone"}])


def test_skip_when_extra_missing():
    with pytest.raises(_SkipSample):
        RolloutRefreshService._build_rollout_payload(
            sample={}, env="MEMORY", task_id=1,
            champion_hotkey="HK", champion_revision="rev",
            tokenizer=_FakeTokenizer(), tokenizer_sig="s", day="d",
        )
