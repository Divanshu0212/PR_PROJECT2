"""Groq backend for `qwen/qwen3.6-27b` (131k context).

Three things about this model and endpoint shape the client:

1. **Cloudflare blocks `urllib`.** Requests through `urllib.request` are rejected
   with `403 / error code: 1010` *before* Groq evaluates the key — a bogus key
   and a valid one fail identically. `httpx` and `curl` pass, so the transport
   is httpx rather than the stdlib used elsewhere in this repo.
2. **It is a reasoning model.** It emits a `<think>…</think>` block inline, which
   breaks both `json_schema` (unsupported by this model) and `json_object`
   (validation fails on the reasoning prefix). Passing `reasoning_format=hidden`
   suppresses the block server-side and yields clean JSON; `strip_reasoning`
   defends against anything that leaks through anyway.
3. **Several keys are available.** They round-robin so parallel workers spread
   load across quotas, and a failing key falls through to the next rather than
   killing the call.

Constrained decoding is therefore *prompt-plus-validation* rather than
grammar-enforced, unlike the Ollama path. Pydantic validation at the call site
is what rejects malformed output — no silent fills.
"""

import itertools
import json
import re
import threading

MODEL = "qwen/qwen3.6-27b"
_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_TIMEOUT_SECONDS = 180

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def load_api_keys(env_text: str) -> list[str]:
    """Read GROQ_API_KEY from .env text. Accepts a JSON array or a bare key."""
    match = re.search(r"GROQ_API_KEY\s*=\s*(.+)", env_text)
    if not match:
        raise ValueError("GROQ_API_KEY not found in env text")
    raw = match.group(1).strip().strip("'\"")
    if raw.startswith("["):
        keys = json.loads(re.search(r"(\[.*?\])", raw, re.S).group(1))
    else:
        keys = [raw]
    keys = [k for k in (k.strip() for k in keys) if k]
    if not keys:
        raise ValueError("GROQ_API_KEY is empty")
    return keys


def load_api_keys_from_file(path: str = ".env") -> list[str]:
    with open(path) as fh:
        return load_api_keys(fh.read())


def strip_reasoning(content: str) -> str:
    """Return the JSON payload from a model response.

    `reasoning_format=hidden` should make this a no-op, but the model still
    occasionally wraps output in a fence or leaks a think block, and a parse
    failure costs a whole benchmark pair.
    """
    text = _THINK.sub("", content).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start > 0:
        text = text[start:]
    return text.strip()


def _httpx_transport(url: str, headers: dict, payload: dict, timeout: int) -> str:
    import httpx

    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.text


class GroqClient:
    """Thread-safe round-robin client over several API keys."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        model: str = MODEL,
        transport=_httpx_transport,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ):
        self._keys = list(api_keys) if api_keys else load_api_keys_from_file()
        self._cycle = itertools.cycle(self._keys)
        self._lock = threading.Lock()
        self.model = model
        self.transport = transport
        self.temperature = temperature
        self.max_tokens = max_tokens

    def next_key(self) -> str:
        with self._lock:  # itertools.cycle is not thread-safe
            return next(self._cycle)

    def complete_json(self, prompt: str, temperature: float | None = None) -> dict:
        """Send `prompt`, return the parsed JSON object.

        Tries each key once before giving up, so a rate-limited or revoked key
        degrades to a retry instead of a failed benchmark pair.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            # Suppress the <think> block server-side; see module docstring.
            "reasoning_format": "hidden",
        }
        errors = []
        for _ in range(len(self._keys)):
            key = self.next_key()
            try:
                raw = self.transport(
                    _ENDPOINT,
                    {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    payload,
                    _TIMEOUT_SECONDS,
                )
                body = json.loads(raw)
                content = body["choices"][0]["message"]["content"]
                return json.loads(strip_reasoning(content))
            except Exception as exc:  # try the next key before failing the call
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError(f"all {len(self._keys)} Groq keys failed: {errors}")
