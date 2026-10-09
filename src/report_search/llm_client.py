"""Small configurable Chat Completions JSON client with fail-closed responses.

No provider, model, or credential is selected implicitly. JSON mode is only a
transport aid: the generation layer still validates both response schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from http.client import HTTPException, IncompleteRead
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

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite JSON value")
        return parsed

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                       parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    # json.loads accepts escaped lone UTF-16 surrogates. Reject them before
    # downstream UTF-8 fingerprints, requests, audit files or output encoding.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeError:
                raise ValueError("Invalid Unicode scalar value") from None
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_response(response, limit):
    """Bound the body and reject incomplete HTTP framing, even for valid JSON.

    HTTPResponse.read(amt) can return a short fixed-length body without raising
    IncompleteRead. A complete JSON prefix is therefore insufficient to prove
    that the response was fully received. Chunked decoding raises separately.
    """
    getheader = getattr(response, "getheader", None)
    length = getheader("Content-Length") if callable(getheader) else None
    transfer = getheader("Transfer-Encoding") if callable(getheader) else None
    if transfer is not None:
        # This client supports one chunked coding. urllib joins duplicate
        # headers here, although HTTPResponse may have decoded only the first.
        if (not isinstance(transfer, str) or transfer.strip().lower() != "chunked"
                or length is not None):
            raise ModelError("invalid_model_response_framing")
    expected = None
    if length is not None and not getattr(response, "chunked", False):
        if not isinstance(length, str) or not length.strip() or any(
                char not in "0123456789" for char in length.strip()):
            raise ModelError("invalid_model_response_framing")
        # Avoid conversion of an attacker-controlled, arbitrarily long integer.
        digits = length.strip().lstrip("0") or "0"
        if len(digits) > len(str(limit)):
            raise ModelError("model_response_too_large")
        expected = int(digits)
        if expected > limit:
            raise ModelError("model_response_too_large")
    raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ModelError("model_response_too_large")
    if expected is not None and len(raw) != expected:
        raise ModelError("incomplete_model_response")
    return raw


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
        try:
            self.base_url.encode("utf-8")
            self.model.encode("utf-8")
        except UnicodeError:
            raise ValueError("API URL and model must contain valid Unicode") from None
        url = urlsplit(self.base_url)
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or url.scheme not in {"https", "http"}):
            raise ValueError("API base URL must have a host and no credentials/query/fragment")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote model endpoints require HTTPS")
        if (not isinstance(self.api_key, str) or not self.api_key.isascii()
                or any(ord(c) < 32 or ord(c) == 127 for c in self.api_key)):
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
        try:
            request = Request(config.base_url.rstrip("/") + "/chat/completions",
                              data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                              headers=headers, method="POST")
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise ModelError("invalid_model_request") from None
        try:
            with build_opener(_NoRedirect()).open(request, timeout=config.timeout_seconds) as response:
                raw = _read_response(response, config.max_response_bytes)
        except HTTPError as error:
            # Do not expose provider bodies, URLs, headers, or credentials.
            raise ModelError(f"model_http_{error.code}") from None
        except IncompleteRead:
            raise ModelError("incomplete_model_response") from None
        except (URLError, TimeoutError, OSError, HTTPException):
            raise ModelError("model_unavailable") from None
        except (ValueError, UnicodeError):
            raise ModelError("invalid_model_request") from None
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
