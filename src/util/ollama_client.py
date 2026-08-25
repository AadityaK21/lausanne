"""
Thin Ollama client.

Speaks raw HTTP rather than depending on the `ollama` package, for three
reasons: the wire API is stable and tiny, it keeps the dependency surface
small on a machine where every dependency is another few hundred MB, and it
lets us set `keep_alive: 0` on *every* call, which the convenience wrappers
do not always expose.

`keep_alive: 0` is not a micro-optimisation here. With 16 GB of system RAM and
8 GB of VRAM, a model left resident between stages is the difference between
a pipeline that runs and one that falls back to shared memory and takes forty
minutes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .logging import get_logger

log = get_logger("ollama")

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 900,
        keep_alive: str | int = 0,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._session = requests.Session()

    # ---------------------------------------------------------------- health

    def is_up(self) -> bool:
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        r = self._session.get(f"{self.host}/api/tags", timeout=15)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def has_model(self, model: str) -> bool:
        available = self.list_models()
        if model in available:
            return True
        # "qwen3:8b" should match a pulled "qwen3:8b-q4_K_M"
        base = model.split(":")[0]
        return any(a.split(":")[0] == base for a in available)

    def resident(self) -> List[Dict[str, Any]]:
        """What is currently loaded in VRAM. Equivalent to `ollama ps`."""
        try:
            r = self._session.get(f"{self.host}/api/ps", timeout=10)
            r.raise_for_status()
            return r.json().get("models", [])
        except requests.RequestException:
            return []

    def unload(self, model: str) -> None:
        """
        Evict a model immediately.

        Called defensively before and after every LLM stage. On a machine this
        size, assuming eviction happened is how you get an OOM three stages
        later.
        """
        try:
            self._session.post(
                f"{self.host}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=30,
            )
            log.debug("unload requested for %s", model)
        except requests.RequestException as exc:
            log.warning("unload of %s failed (non-fatal): %s", model, exc)

    def unload_all(self) -> None:
        for m in self.resident():
            self.unload(m.get("name", ""))

    # ------------------------------------------------------------- generate

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        format_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 1536,
        think: bool = False,
        extra_options: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ) -> str:
        """
        One chat completion. Returns the raw message content.

        `think=False` matters for Qwen3-family models, which otherwise emit a
        reasoning block that both burns the output budget and breaks strict
        JSON parsing.
        """
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            # Quantised KV cache: halves cache memory at negligible quality
            # cost, which is what makes an 8K context fit alongside the weights.
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        }
        if extra_options:
            options.update(extra_options)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
            "think": think,
        }
        if format_schema is not None:
            payload["format"] = format_schema

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                t0 = time.perf_counter()
                r = self._session.post(
                    f"{self.host}/api/chat", json=payload, timeout=self.timeout
                )
                if r.status_code == 400 and "think" in r.text.lower():
                    # Older Ollama builds reject the `think` key outright.
                    payload.pop("think", None)
                    continue
                r.raise_for_status()
                data = r.json()
                content = data.get("message", {}).get("content", "")
                log.debug(
                    "chat %s: %d chars in %.1fs (eval %s tok)",
                    model,
                    len(content),
                    time.perf_counter() - t0,
                    data.get("eval_count", "?"),
                )
                return content
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < retries:
                    wait = 2 ** attempt
                    log.warning(
                        "chat failed (attempt %d/%d): %s - retrying in %ds",
                        attempt + 1, retries + 1, exc, wait,
                    )
                    time.sleep(wait)
                    continue
        raise OllamaError(
            f"Ollama chat failed after {retries + 1} attempts: {last_exc}"
        )

    def chat_structured(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Chat constrained to a Pydantic schema.

        Ollama enforces the JSON grammar at decode time, so malformed JSON is
        essentially impossible. Semantic violations (an id that does not exist)
        still are possible, which is why callers must validate ids separately.
        """
        schema = response_model.model_json_schema()
        content = self.chat(
            model, messages, format_schema=schema, **kwargs
        )
        try:
            return response_model.model_validate_json(content)
        except ValidationError:
            # Rare: a model emits a JSON-looking prefix/suffix around the object.
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end > start:
                return response_model.model_validate_json(content[start:end + 1])
            raise OllamaError(
                f"Model returned unparseable output for "
                f"{response_model.__name__}: {content[:400]!r}"
            )


def preflight(client: OllamaClient, model: str) -> None:
    """
    Fail fast with an actionable message rather than deep inside a map loop.
    """
    if "configure-me.invalid" in client.host:
        raise OllamaError(
            "This tier has no endpoint configured.\n"
            "The hosted tier ships without a default on purpose: a localhost "
            "default would mean selecting the 70B tier silently ran your local "
            "8B model and labelled the result as 70B.\n"
            "Set tiers.hosted.host in config.yaml, or export OLLAMA_HOST."
        )
    if not client.is_up():
        raise OllamaError(
            f"Cannot reach Ollama at {client.host}. Start it "
            f"(the tray icon on Windows, or `ollama serve`) and retry."
        )
    if not client.has_model(model):
        raise OllamaError(
            f"Model {model!r} is not pulled. Run:  ollama pull {model}\n"
            f"Currently available: {', '.join(client.list_models()) or '(none)'}"
        )
