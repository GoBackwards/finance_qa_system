import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from report_search.llm_client import ChatCompletionsClient, ModelConfig, ModelError, strict_json


class Response(io.BytesIO):
    pass


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
                  HTTPError("https://secret-test-key", 401, "secret-test-key", {}, None)]
        for error in errors:
            with patch("report_search.llm_client.build_opener") as opener:
                opener.return_value.open.side_effect = error
                with self.assertRaises(ModelError) as raised:
                    self.client().complete_json([])
                self.assertNotIn("secret-test-key", str(raised.exception))

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
        for data in ('{"a":1,"a":2}', '{"a":Infinity}', '[]'):
            with self.assertRaises(ValueError):
                strict_json(data)


if __name__ == "__main__":
    unittest.main()
