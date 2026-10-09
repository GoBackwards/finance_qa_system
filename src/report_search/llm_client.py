"""Small configurable Chat Completions JSON client with fail-closed responses.

No provider, model, or credential is selected implicitly. JSON mode is only a
transport aid: the generation layer still validates both response schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ModelError(RuntimeError):
    """Sanitized provider/configuration failure; never contains credentials."""


def strict_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Non-finite JSON value")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 45.0
    max_output_tokens: int = 2048
    token_parameter: str = "max_completion_tokens"
    max_response_bytes: int = 2_000_000

    def __post_init__(self):
        if not isinstance(self.base_url, str) or not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Set an explicit API base URL and model")
        url = urlsplit(self.base_url)
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or url.scheme not in {"https", "http"}):
            raise ValueError("API base URL must have a host and no credentials/query/fragment")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote model endpoints require HTTPS")
        if not isinstance(self.api_key, str) or any(c in self.api_key for c in "\r\n"):
            raise ValueError("Invalid API credential")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        for value in (self.max_output_tokens, self.max_response_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Token and response limits must be positive integers")
        if self.token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported completion token parameter")

    @classmethod
    def from_env(cls, *, verifier=False):
        # The verifier may use a different model on the same configured endpoint.
        base = os.getenv("FINANCE_QA_BASE_URL", "")
        model = os.getenv("FINANCE_QA_MODEL", "")
        if verifier:
            model = os.getenv("FINANCE_QA_VERIFIER_MODEL", "") or model
        if not base or not model:
            raise ModelError("model_not_configured")
        try:
            return cls(base_url=base, model=model, api_key=os.getenv("FINANCE_QA_API_KEY", ""),
                       token_parameter=os.getenv("FINANCE_QA_TOKEN_PARAMETER", "max_completion_tokens"))
        except ValueError:
            raise ModelError("invalid_model_configuration") from None


class ChatCompletionsClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def identity(self):
        return {"backend": "openai_compatible_chat_completions", "model": self.config.model,
                "endpoint_host": urlsplit(self.config.base_url).hostname,
                "json_mode": True, "max_output_tokens": self.config.max_output_tokens}

    def complete_json(self, messages: list[dict]) -> dict:
        config = self.config
        payload = {"model": config.model, "messages": messages, "stream": False,
                   "response_format": {"type": "json_object"},
                   config.token_parameter: config.max_output_tokens}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.api_key:
            headers["Authorization"] = "Bearer " + config.api_key
        request = Request(config.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=config.timeout_seconds) as response:
                raw = response.read(config.max_response_bytes + 1)
                if len(raw) > config.max_response_bytes:
                    raise ModelError("model_response_too_large")
        except HTTPError as error:
            # Do not expose provider bodies, URLs, headers, or credentials.
            raise ModelError(f"model_http_{error.code}") from None
        except (URLError, TimeoutError, OSError):
            raise ModelError("model_unavailable") from None
        try:
            envelope = strict_json(raw.decode("utf-8"))
            choices = envelope.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ModelError("invalid_model_choices")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise ModelError("incomplete_model_response")
            message = choice["message"]
            if message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
                raise ModelError("model_refusal_or_tool_call")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ModelError("empty_model_response")
            return strict_json(content)
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
            raise ModelError("invalid_model_json") from None
