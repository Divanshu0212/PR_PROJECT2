"""Groq backend: key rotation and response parsing.

No network here. The transport is injected so rotation and parsing are pinned
deterministically; live reachability is covered by the integration tests.
"""

import json

import pytest

from rho.llm.groq import GroqClient, load_api_keys, strip_reasoning


def test_load_api_keys_parses_json_array():
    keys = load_api_keys('GROQ_API_KEY=["a","b","c"]')
    assert keys == ["a", "b", "c"]


def test_load_api_keys_accepts_bare_single_key():
    assert load_api_keys("GROQ_API_KEY=solo") == ["solo"]


def test_load_api_keys_errors_when_absent():
    with pytest.raises(ValueError):
        load_api_keys("SOMETHING_ELSE=1")


def test_keys_rotate_round_robin():
    """Each concurrent worker must land on a different key to spread the quota."""
    client = GroqClient(api_keys=["k1", "k2", "k3"], transport=lambda *a, **k: "{}")
    assert [client.next_key() for _ in range(4)] == ["k1", "k2", "k3", "k1"]


def test_strip_reasoning_removes_think_block():
    """Qwen3.6 emits <think> inline; JSON starts after it."""
    raw = '<think>\nplanning...\n</think>\n{"skills": ["Python"]}'
    assert json.loads(strip_reasoning(raw)) == {"skills": ["Python"]}


def test_strip_reasoning_extracts_fenced_json():
    raw = 'Here you go:\n```json\n{"skills": ["SQL"]}\n```'
    assert json.loads(strip_reasoning(raw)) == {"skills": ["SQL"]}


def test_strip_reasoning_passes_through_clean_json():
    assert json.loads(strip_reasoning('{"a": 1}')) == {"a": 1}


def test_complete_json_returns_parsed_payload():
    def fake_transport(url, headers, payload, timeout):
        return json.dumps(
            {"choices": [{"message": {"content": '{"skills":["Go"]}'}}]}
        )

    client = GroqClient(api_keys=["k1"], transport=fake_transport)
    assert client.complete_json("prompt") == {"skills": ["Go"]}


def test_complete_json_retries_on_failure_with_next_key():
    """A dead or rate-limited key must not fail the call outright."""
    calls = []

    def flaky(url, headers, payload, timeout):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            raise RuntimeError("429 rate limit")
        return json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]})

    client = GroqClient(api_keys=["k1", "k2"], transport=flaky)
    assert client.complete_json("prompt") == {"ok": True}
    assert calls == ["Bearer k1", "Bearer k2"]


def test_complete_json_raises_after_exhausting_keys():
    def always_fail(url, headers, payload, timeout):
        raise RuntimeError("boom")

    client = GroqClient(
        api_keys=["k1", "k2"], transport=always_fail, rounds=1, backoff=0
    )
    with pytest.raises(RuntimeError):
        client.complete_json("prompt")


def test_complete_json_retries_rotation_more_than_once():
    """A 429 burst hits every key at once; one pass through them is not enough."""
    calls = []

    def rate_limited_then_ok(url, headers, payload, timeout):
        calls.append(headers["Authorization"])
        if len(calls) <= 2:  # both keys 429 on the first rotation
            raise RuntimeError("429 Too Many Requests")
        return json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]})

    client = GroqClient(
        api_keys=["k1", "k2"], transport=rate_limited_then_ok, rounds=3, backoff=0
    )
    assert client.complete_json("prompt") == {"ok": True}
    assert len(calls) == 3  # recovered on the second rotation


def test_parse_duration_handles_groq_formats():
    from rho.llm.groq import _parse_duration

    assert _parse_duration("5") == 5.0
    assert _parse_duration("33.517s") == pytest.approx(33.517)
    assert _parse_duration("1m33.5s") == pytest.approx(93.5)


def test_rate_limited_waits_for_server_reset_not_blind_backoff():
    """TPM limits are account-wide, so rotating keys alone cannot recover."""
    from rho.llm.groq import RateLimited

    slept, calls = [], []

    def limited_once(url, headers, payload, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise RateLimited(12.0, "429")
        return json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]})

    client = GroqClient(api_keys=["k1", "k2"], transport=limited_once, max_wait=70)
    import rho.llm.groq as mod

    original = mod.time.sleep
    mod.time.sleep = slept.append
    try:
        assert client.complete_json("p") == {"ok": True}
    finally:
        mod.time.sleep = original
    assert slept and 12.0 <= slept[0] <= 13.0  # server's reset + jitter


def test_rate_limit_wait_is_capped():
    from rho.llm.groq import RateLimited

    slept = []

    def always_limited(url, headers, payload, timeout):
        raise RateLimited(3600.0, "429")

    client = GroqClient(
        api_keys=["k1"], transport=always_limited, rounds=2, max_wait=70
    )
    import rho.llm.groq as mod

    original = mod.time.sleep
    mod.time.sleep = slept.append
    try:
        with pytest.raises(RuntimeError):
            client.complete_json("p")
    finally:
        mod.time.sleep = original
    assert all(s <= 71 for s in slept)  # never blocks for the full hour
