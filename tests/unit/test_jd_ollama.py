"""Ollama-backed JD analysis — structured extraction without a CUDA GPU."""

import json

import pytest

from rho.jd.ollama import _build_payload, _parse_response


def test_payload_pins_temperature_and_schema():
    payload = _build_payload("need python", model="gemma3:4b")
    assert payload["model"] == "gemma3:4b"
    assert payload["stream"] is False
    # Determinism is required for the reproducibility claim.
    assert payload["options"]["temperature"] == 0
    # Constrained decoding: the JD schema is enforced by the server.
    assert payload["format"]["properties"]["requirements"]["type"] == "array"
    assert "need python" in payload["messages"][0]["content"]


def test_parse_response_maps_to_jd_schema():
    raw = {
        "message": {
            "content": json.dumps(
                {
                    "reasoning": "r",
                    "title": "Backend Engineer",
                    "requirements": [
                        {"text": "Python", "kind": "skill", "priority": "must", "years": 3},
                        {"text": "Docker", "kind": "tool", "priority": "nice"},
                    ],
                }
            )
        }
    }
    js = _parse_response(raw)
    assert js.title == "Backend Engineer"
    assert [r.text for r in js.requirements] == ["Python", "Docker"]
    assert js.requirements[0].priority == "must"
    assert js.requirements[0].years == 3
    assert js.requirements[1].years is None


def test_parse_response_rejects_malformed_json():
    with pytest.raises(ValueError):
        _parse_response({"message": {"content": "not json"}})


def test_parse_response_drops_requirements_missing_required_fields():
    """A malformed item is dropped, not silently defaulted into a fake requirement."""
    raw = {
        "message": {
            "content": json.dumps(
                {
                    "reasoning": "r",
                    "requirements": [
                        {"text": "Python", "kind": "skill", "priority": "must"},
                        {"kind": "skill", "priority": "must"},
                    ],
                }
            )
        }
    }
    assert [r.text for r in _parse_response(raw).requirements] == ["Python"]
