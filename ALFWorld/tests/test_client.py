import json
import io
import unittest
import urllib.error
from unittest.mock import patch

from continual_alfworld.client import OpenAICompatibleClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ).encode("utf-8")


class ClientTests(unittest.TestCase):
    def test_versioned_cloud_base_and_thinking_disabled_request(self):
        client = OpenAICompatibleClient(
            "https://example.invalid/v1",
            "test-secret",
            "test-model",
            extra_body={"thinking": {"type": "disabled"}},
        )
        captured = {}

        def fake_open(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.build_opener") as build_opener:
            build_opener.return_value.open.side_effect = fake_open
            self.assertEqual(
                client.complete([{"role": "user", "content": "Hi"}], "action"), "ok"
            )
        self.assertEqual(captured["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(captured["body"]["thinking"], {"type": "disabled"})
        self.assertEqual(client.stats["action"]["completion_tokens"], 2)

    def test_extra_body_cannot_override_standard_request_fields(self):
        client = OpenAICompatibleClient(
            "https://example.invalid",
            "",
            "test-model",
            extra_body={"model": "other-model"},
        )
        with self.assertRaisesRegex(ValueError, "overrides standard fields"):
            client.complete([{"role": "user", "content": "Hi"}], "action")

    def test_403_reports_redacted_provider_detail_without_retry(self):
        client = OpenAICompatibleClient(
            "https://example.invalid", "test-secret", "test-model"
        )
        body = json.dumps(
            {
                "error": {
                    "type": "permission_error",
                    "code": "account_blocked",
                    "message": "Key test-secret is not permitted",
                }
            }
        ).encode("utf-8")
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            403,
            "Forbidden",
            {},
            io.BytesIO(body),
        )
        with patch("urllib.request.build_opener") as build_opener:
            build_opener.return_value.open.side_effect = error
            with self.assertRaises(RuntimeError) as raised:
                client.complete([{"role": "user", "content": "Hi"}], "action")
            self.assertEqual(build_opener.return_value.open.call_count, 1)
        self.assertIn("account_blocked", str(raised.exception))
        self.assertNotIn("test-secret", str(raised.exception))


class AccountingFailureTests(unittest.TestCase):
    def client(self):
        return OpenAICompatibleClient(
            "https://example.invalid/v1", "test-secret", "fixed-model", retries=3
        )

    def test_unknown_transport_failure_is_not_retried_and_has_durable_start(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            client = self.client()
            client.journal_path = Path(directory) / "requests.jsonl"
            client.active_task_key = "task-1"
            with patch("urllib.request.build_opener") as build:
                build.return_value.open.side_effect = urllib.error.URLError(
                    "transport failure"
                )
                with self.assertRaises(RuntimeError):
                    client.complete([], "action")
                self.assertEqual(build.return_value.open.call_count, 1)
            events = [
                json.loads(line)
                for line in client.journal_path.read_text().splitlines()
            ]
            self.assertEqual(
                [e["event"] for e in events], ["request_start", "request_error"]
            )
            self.assertEqual(events[0]["task"], "task-1")
            self.assertEqual(client.stats["action"]["calls"], 1)

    def test_explicit_rate_limit_has_bounded_attempt_accounting(self):
        client = self.client()
        error = urllib.error.HTTPError(
            "https://example.invalid", 429, "Limited", {}, io.BytesIO(b"{}")
        )
        with patch("urllib.request.build_opener") as build, patch("time.sleep"):
            build.return_value.open.side_effect = [error, FakeResponse()]
            self.assertEqual(client.complete([], "action"), "ok")
        self.assertEqual(client.stats["action"]["calls"], 2)
        self.assertEqual(client.stats["action"]["errors"], 1)
        self.assertEqual(client.stats["action"]["completion_tokens"], 2)

    def test_server_error_with_uncertain_usage_is_not_retried(self):
        client = self.client()
        error = urllib.error.HTTPError(
            "https://example.invalid", 503, "Fault", {}, io.BytesIO(b"{}")
        )
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = error
            with self.assertRaises(RuntimeError):
                client.complete([], "action")
            self.assertEqual(build.return_value.open.call_count, 1)

    def test_missing_malformed_or_over_budget_usage_and_empty_reply_fail(self):
        cases = [
            {},
            {"prompt_tokens": 3, "completion_tokens": True},
            {"prompt_tokens": 3, "completion_tokens": -1},
            {"prompt_tokens": 3, "completion_tokens": 129},
        ]
        for usage in cases:
            with self.subTest(usage=usage):
                client = self.client()
                with patch.object(
                    client,
                    "_request",
                    return_value={
                        "usage": usage,
                        "choices": [{"message": {"content": "ok"}}],
                    },
                ):
                    with self.assertRaises(RuntimeError):
                        client.complete([], "action", max_tokens=128)
        client = self.client()
        with patch.object(
            client,
            "_request",
            return_value={
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "choices": [],
            },
        ):
            with self.assertRaises(RuntimeError):
                client.complete([], "action")
