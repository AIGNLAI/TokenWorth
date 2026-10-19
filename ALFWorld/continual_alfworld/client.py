"""Small OpenAI-compatible client with complete cost accounting."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence
from tokenworth_core.runtime import append_event


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 180,
        retries: int = 3,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.extra_body = dict(extra_body or {})
        self.journal_path = None
        self.active_task_key = None
        self._purpose = "action"
        if not 1 <= self.retries <= 10:
            raise ValueError("model.retries must be bounded in [1,10]")
        self.stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_seconds": 0.0,
                "errors": 0,
            }
        )

    def _request(
        self, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        # Both bare API hosts and OpenAI-SDK-style bases ending in /v1 are
        # accepted; avoid accidentally requesting /v1/v1/chat/completions.
        request_path = (
            path[3:]
            if self.base_url.endswith("/v1") and path.startswith("/v1/")
            else path
        )
        request = urllib.request.Request(
            self.base_url + request_path, data=body, headers=headers
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        last_error: Optional[Exception] = None
        is_generation = path.endswith("/chat/completions")
        for attempt in range(self.retries):
            if is_generation:
                self.stats[self._purpose]["calls"] += 1
                append_event(
                    self.journal_path,
                    {
                        "event": "request_start",
                        "task": self.active_task_key,
                        "purpose": self._purpose,
                        "attempt": attempt + 1,
                    },
                )
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if is_generation:
                    append_event(
                        self.journal_path,
                        {
                            "event": "request_response",
                            "task": self.active_task_key,
                            "purpose": self._purpose,
                            "usage": result.get("usage"),
                        },
                    )
                return result
            except urllib.error.HTTPError as error:
                # HTTPError is also a URLError. Preserve the provider's error
                # details instead of retrying an authorization failure and
                # reporting only its status line.
                raw = error.read(4096).decode("utf-8", "replace")
                try:
                    payload_error = json.loads(raw).get("error", {})
                except (ValueError, AttributeError):
                    payload_error = {}
                if not isinstance(payload_error, dict):
                    payload_error = {}
                details = []
                for field in ("type", "code", "message"):
                    value = (
                        str(payload_error.get(field) or "").replace(
                            self.api_key, "<redacted>"
                        )
                        if self.api_key
                        else str(payload_error.get(field) or "")
                    )
                    if value:
                        details.append(
                            "{}={}".format(field, value.replace("\n", " ")[:300])
                        )
                last_error = "HTTP {} {}".format(error.code, "; ".join(details)).strip()
                if is_generation:
                    self.stats[self._purpose]["errors"] += 1
                    append_event(
                        self.journal_path,
                        {
                            "event": "request_error",
                            "task": self.active_task_key,
                            "status": error.code,
                        },
                    )
                if error.code != 429:
                    break
                if attempt + 1 < self.retries:
                    time.sleep(min(4, 2**attempt))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if is_generation:
                    self.stats[self._purpose]["errors"] += 1
                    append_event(
                        self.journal_path,
                        {
                            "event": "request_error",
                            "task": self.active_task_key,
                            "error_type": type(error).__name__,
                        },
                    )
                break
        raise RuntimeError("OpenAI-compatible request failed: {}".format(last_error))

    def resolve_model(self) -> str:
        if self.model != "auto":
            return self.model
        response = self._request("/v1/models")
        self.model = str(response["data"][0]["id"])
        return self.model

    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
    ) -> str:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        model = self.resolve_model()
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        reserved = set(payload) & set(self.extra_body)
        if reserved:
            raise ValueError(
                "model.extra_body overrides standard fields: "
                + ", ".join(sorted(reserved))
            )
        payload.update(self.extra_body)
        started = time.perf_counter()
        stats = self.stats[purpose]
        self._purpose = purpose
        try:
            response = self._request("/v1/chat/completions", payload)
        except Exception:
            stats["latency_seconds"] += time.perf_counter() - started
            raise
        stats["latency_seconds"] += time.perf_counter() - started
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                stats["errors"] += 1
                raise RuntimeError("Missing/invalid provider usage: " + key)
            stats[key] += value
        if usage["completion_tokens"] < 1 or usage["completion_tokens"] > max_tokens:
            stats["errors"] += 1
            raise RuntimeError(
                "Provider output token count violates the request budget"
            )
        choices = response.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(content, str) or not content.strip():
            stats["errors"] += 1
            raise RuntimeError("Model returned an empty or malformed response")
        return content

    def snapshot_stats(self) -> Dict[str, Dict[str, float]]:
        return {purpose: dict(values) for purpose, values in self.stats.items()}


def parse_json_object(text: str) -> Dict[str, Any]:
    fenced = text
    if "```json" in text:
        fenced = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        fenced = text.split("```", 1)[1].split("```", 1)[0]
    start, end = fenced.find("{"), fenced.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(fenced[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value
