import io
from http.client import BadStatusLine, HTTPResponse
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from report_search.llm_client import ChatCompletionsClient, ModelConfig, ModelError, strict_json
from report_search.generation import generate_answer


class Response(io.BytesIO):
    pass


class WireSocket:
    """Exercise real stdlib HTTP framing without a listening socket/network."""

    def __init__(self, wire):
        self.wire = wire

    def makefile(self, *args):
        return io.BytesIO(self.wire)


def envelope(content='{"status":"abstain","claims":[]}', **changes):
    choice = {"finish_reason": "stop", "message": {"content": content}}
    choice.update(changes)
    return json.dumps({"choices": [choice]}).encode()


class ClientTests(unittest.TestCase):
    def client(self, **options):
        return ChatCompletionsClient(ModelConfig("https://provider.example/v1", "chosen-model",
                                               api_key="secret-test-key", **options))

    def invoke(self, response, **options):
        with patch("report_search.llm_client.build_opener") as opener:
            opener.return_value.open.return_value = Response(response)
            result = self.client(**options).complete_json([{"role": "user", "content": "JSON please"}])
            return result, opener.return_value.open.call_args

    def test_wire_format_and_explicit_limits(self):
        result, call = self.invoke(envelope())
        self.assertEqual(result, {"status": "abstain", "claims": []})
        request = call.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://provider.example/v1/chat/completions")
        self.assertEqual(body["model"], "chosen-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_completion_tokens"], 2048)
        self.assertFalse(body["stream"])
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-test-key")
        self.assertNotIn("secret-test-key", repr(self.client().config))

    def test_legacy_compatible_token_parameter(self):
        _, call = self.invoke(envelope(), token_parameter="max_tokens")
        self.assertIn("max_tokens", json.loads(call.args[0].data))

    def test_truncation_refusal_tools_and_malformed_json_fail(self):
        cases = [envelope(finish_reason="length"), envelope(content="not JSON"),
                 envelope(content='{"x":1,"x":2}'), envelope(content='{"x":NaN}'),
                 envelope(message={"content": "{}", "refusal": "cannot"}),
                 envelope(message={"content": "{}", "tool_calls": [{}]}),
                 b'{"choices":[]}', b'[]', b'invalid',
                 envelope(message={"content": ["not a string"]})]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(ModelError):
                self.invoke(response)

    def test_oversized_response(self):
        with self.assertRaisesRegex(ModelError, "too_large"):
            self.invoke(envelope(), max_response_bytes=5)

    def test_network_error_does_not_expose_credentials_or_body(self):
        errors = [URLError("secret-test-key"),
                  BadStatusLine("secret-test-key"),
                  HTTPError("https://secret-test-key", 401, "secret-test-key", {}, None)]
        for error in errors:
            with patch("report_search.llm_client.build_opener") as opener:
                opener.return_value.open.side_effect = error
                with self.assertRaises(ModelError) as raised:
                    self.client().complete_json([])
                self.assertNotIn("secret-test-key", str(raised.exception))

    def invoke_wire(self, headers, body, **options):
        response = HTTPResponse(WireSocket(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + body))
        response.begin()
        with patch("report_search.llm_client.build_opener") as opener:
            opener.return_value.open.return_value = response
            return self.client(**options).complete_json([])

    def test_short_content_length_rejected_even_when_body_is_valid_json(self):
        body = envelope()
        with self.assertRaisesRegex(ModelError, "^incomplete_model_response$"):
            self.invoke_wire(f"Content-Length: {len(body) + 20}\r\n".encode(), body)

    def test_unterminated_chunked_response_is_sanitized(self):
        body = envelope()
        chunk = f"{len(body):x}\r\n".encode() + body + b"\r\n"
        with self.assertRaisesRegex(ModelError, "^incomplete_model_response$"):
            self.invoke_wire(b"Transfer-Encoding: chunked\r\n", chunk)

    def test_partial_provider_response_becomes_public_abstention(self):
        body = envelope()
        chunk = f"{len(body):x}\r\n".encode() + body + b"\r\n"
        response = HTTPResponse(WireSocket(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk))
        response.begin()
        source, verifier = Mock(), Mock()
        source.build.return_value = {"source_verified": True, "evidence": {"E1": {"text": "Revenue was $10 million."}}}
        with patch("report_search.llm_client.build_opener") as opener:
            opener.return_value.open.return_value = response
            result = generate_answer("What was revenue?", [{"chunk_id": "c1", "reranker_score": 7.0}],
                                     threshold=6.0, source_store=source,
                                     generator=self.client(), verifier=verifier)
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertEqual(result["reason"], "incomplete_model_response")
        self.assertEqual(result["citations"], [])
        verifier.complete_json.assert_not_called()

    def test_complete_http_framing_preserves_compatible_responses(self):
        body = envelope()
        chunk = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
        for headers, wire_body in ((f"Content-Length: {len(body)}\r\n".encode(), body),
                                   (b"Transfer-Encoding: chunked\r\n", chunk),
                                   (b"Connection: close\r\n", body)):
            with self.subTest(headers=headers):
                self.assertEqual(self.invoke_wire(headers, wire_body), {"status": "abstain", "claims": []})

    def test_invalid_and_oversized_declared_lengths_are_rejected(self):
        for length in (b"garbage", b"-1", b"1, 1"):
            with self.subTest(length=length), self.assertRaisesRegex(ModelError, "framing"):
                self.invoke_wire(b"Content-Length: " + length + b"\r\n", envelope())
        for length in (b"2000001", b"9" * 5000):
            with self.subTest(length=length[:10]), self.assertRaisesRegex(ModelError, "too_large"):
                self.invoke_wire(b"Content-Length: " + length + b"\r\n", envelope())

    def test_conflicting_and_duplicate_transfer_framing_is_rejected(self):
        body = envelope()
        chunk = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
        headers = (b"Content-Length: 1\r\nTransfer-Encoding: chunked\r\n",
                   b"Transfer-Encoding: chunked\r\nTransfer-Encoding: gzip\r\n",
                   b"Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n",
                   b"Transfer-Encoding: gzip, chunked\r\n")
        for header in headers:
            with self.subTest(header=header), self.assertRaisesRegex(ModelError, "^invalid_model_response_framing$"):
                self.invoke_wire(header, chunk)

    def test_invalid_request_never_reaches_transport_or_exposes_input(self):
        circular = []
        circular.append(circular)
        cases = [[{"role": "user", "content": "secret-test-key\ud800"}],
                 [{"content": float("nan")}], [{"content": object()}], circular]
        for messages in cases:
            with self.subTest(kind=type(messages[0]).__name__), patch("report_search.llm_client.build_opener") as opener:
                with self.assertRaisesRegex(ModelError, "^invalid_model_request$"):
                    self.client().complete_json(messages)
                opener.assert_not_called()

    def test_invalid_credentials_are_rejected_before_transport(self):
        for key in ("secret-test-key\u2603", "secret-test-key\t", "secret-test-key\x00", "secret-test-key\x7f"):
            with self.subTest(key_suffix=repr(key[-1:])), self.assertRaisesRegex(ValueError, "^Invalid API credential$"):
                ModelConfig("https://provider.example/v1", "chosen-model", api_key=key)

    def test_endpoint_and_config_validation(self):
        for url in ("http://remote.example/v1", "https://key@host/v1", "https://host/v1?key=x", "", "file:///tmp"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ModelConfig(url, "m")
        self.assertEqual(ModelConfig("http://127.0.0.1:8080/v1", "m").model, "m")
        with self.assertRaises(ValueError):
            self.client(timeout_seconds=float("nan"))

    def test_environment_has_no_implicit_provider(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ModelError, "not_configured"):
            ModelConfig.from_env()

    def test_json_rejects_duplicate_and_nonfinite_keys(self):
        for data in ('{"a":1,"a":2}', '{"a":Infinity}', '{"a":1e999}', '{"a":[-1e999]}', '[]'):
            with self.assertRaises(ValueError):
                strict_json(data)

    def test_numeric_overflow_is_rejected_in_envelope_and_content(self):
        for response in (envelope(content='{"a":1e999}'),
                         envelope()[:-1] + b',"usage":{"tokens":1e999}}'):
            with self.assertRaisesRegex(ModelError, "^invalid_model_json$"):
                self.invoke(response)
        self.assertEqual(strict_json('{"a":1e100}'), {"a": 1e100})

    def test_json_rejects_lone_surrogates_in_keys_and_nested_values(self):
        for data in (r'{"text":"\ud800"}', r'{"text":"\udfff"}',
                     r'{"\ud800":"value"}', r'{"nested":[{"text":"x\udfff"}]}'):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "^Invalid Unicode scalar value$"):
                strict_json(data)
        valid = strict_json(r'{"text":"营收 café \ud83d\udcca","nested":[{"ключ":"значение"}]}')
        self.assertEqual(valid, {"text": "营收 café 📊", "nested": [{"ключ": "значение"}]})

    def test_provider_surrogates_fail_at_both_json_boundaries(self):
        for body in (envelope(content=r'{"text":"\ud800"}'),
                     envelope()[:-1] + b',"metadata":"\\ud800"}'):
            with self.subTest(body=body), self.assertRaisesRegex(ModelError, "^invalid_model_json$"):
                self.invoke(body)
        result, _ = self.invoke(envelope(content=r'{"text":"营收 \ud83d\udcca"}'))
        self.assertEqual(result, {"text": "营收 📊"})

    def test_model_identity_must_be_utf8_encodable(self):
        for url, model in (("https://provider.example/v1", "model\ud800"),
                           ("https://provider.example/\udfff", "chosen-model")):
            with self.subTest(model=repr(model)), self.assertRaisesRegex(ValueError, "^API URL and model must contain valid Unicode$"):
                ModelConfig(url, model)
        identity = ChatCompletionsClient(ModelConfig("https://provider.example/v1", "模型-📊")).identity
        self.assertEqual(json.loads(json.dumps(identity, ensure_ascii=False).encode("utf-8"))["model"], "模型-📊")


if __name__ == "__main__":
    unittest.main()
