"""Exercise usage accounting and total task budgets through the SDK HTTP transport."""
import json

import pytest

from src.language_models.instance.openai_language_model import OpenaiLanguageModel
from src.callbacks.instance.lifelong_memory_utils import run_single_prompt
from src.typings import (
    ChatHistory, ChatHistoryItem, Role, GenerationBudgetExceeded,
    LanguageModelUnknownException, LanguageModelContextLimitException,
)


@pytest.fixture
def endpoint():
    import httpx
    state = {"requests": [], "usage": True, "content": "usable result", "http_status": 200, "tokens": None}

    def handle(request):
        body = json.loads(request.content)
        state["requests"].append(body)
        status = state["http_status"]
        if status != 200:
            payload = {"error": {"message": "fixture outage", "type": "server_error"}}
        else:
            tokens = state["tokens"] if state["tokens"] is not None else min(body.get("max_tokens", 3), 3)
            payload = {"id": "fixture", "object": "chat.completion", "created": 0,
                       "model": "fixture", "choices": [{"index": 0, "message": {"role": "assistant", "content": state["content"]}, "finish_reason": "stop"}]}
            if state["usage"]:
                payload["usage"] = {"prompt_tokens": 4, "completion_tokens": tokens, "total_tokens": 4 + tokens}
        if state.get("omit_total") and "usage" in payload:
            payload["usage"].pop("total_tokens")
        if state.get("inspect_request"):
            state["inspect_request"]()
        return httpx.Response(status, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        yield state, client


def model_for(endpoint, tmp_path, **kwargs):
    from openai import OpenAI
    model = OpenaiLanguageModel("fixture", {"user": "user", "agent": "assistant"}, api_key="fixture", base_url="http://fixture/v1", trust_env=False, **kwargs)
    model.client.close()
    model.client = OpenAI(api_key="fixture", base_url="http://fixture/v1", http_client=endpoint[1], max_retries=0)
    model.configure_audit(output_dir=str(tmp_path), model_alias="backbone", run_id="fixture-run")
    return model


def history():
    chat = ChatHistory()
    chat.inject(ChatHistoryItem(role=Role.USER, content="test prompt"))
    return chat


def ledger(tmp_path):
    return [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]


def test_agent_and_auxiliary_share_total_budget(endpoint, tmp_path):
    model = model_for(endpoint, tmp_path)
    model.begin_sample("0", 5)
    assert model.inference([history()], {"max_tokens": 128})[0].content
    assert run_single_prompt(model, "memory extraction", max_tokens=128)
    assert run_single_prompt(model, "another extraction", max_tokens=128) == ""
    with pytest.raises(GenerationBudgetExceeded):
        model.inference([history()], {"max_tokens": 128})
    assert [request["max_tokens"] for request in endpoint[0]["requests"]] == [5, 2]
    usage = ledger(tmp_path)
    assert sum(item["completion_tokens"] for item in usage) == 5
    assert [item["phase"] for item in usage[:2]] == ["agent", "auxiliary"]
    assert model.audit_failure is None
    model.begin_sample("1", 5)
    model.inference([history()])
    assert endpoint[0]["requests"][-1]["max_tokens"] == 5


@pytest.mark.parametrize("failure", ["missing_usage", "empty", "overshoot", "outage"])
def test_unverifiable_execution_fails_instead_of_training(endpoint, tmp_path, failure):
    state = endpoint[0]
    if failure == "missing_usage": state["usage"] = False
    if failure == "empty": state["content"] = ""
    if failure == "overshoot": state["tokens"] = 6
    if failure == "outage": state["http_status"] = 503
    model = model_for(endpoint, tmp_path, max_retries=3)
    model.begin_sample("0", 5)
    with pytest.raises(LanguageModelUnknownException):
        model.inference([history()])
    assert model.audit_failure
    assert len(state["requests"]) == 1  # No opaque SDK/server-error retry.
    assert len(ledger(tmp_path)) == 1


def test_prompt_guard_rejects_before_billable_request(endpoint, tmp_path):
    model = model_for(endpoint, tmp_path, maximum_prompt_token_count=10)
    model.begin_sample("0", 5)
    with pytest.raises(LanguageModelContextLimitException):
        model.inference([history()])
    assert not endpoint[0]["requests"]
    assert ledger(tmp_path)[0]["attempt"] == 0
    assert model.audit_failure is None


def test_total_is_derived_from_reported_input_and_output(endpoint, tmp_path):
    endpoint[0]["omit_total"] = True
    model = model_for(endpoint, tmp_path)
    model.begin_sample("0", 5)
    model.inference([history()])
    entry = ledger(tmp_path)[0]
    assert entry["total_tokens"] == entry["prompt_tokens"] + entry["completion_tokens"]
    assert entry["total_tokens_derived"] is True
    assert model.audit_failure is None


def test_request_start_is_durable_before_http_transport(endpoint, tmp_path):
    model = model_for(endpoint, tmp_path)
    model.begin_sample("0", 5)
    def inspect():
        starts = [json.loads(line) for line in (tmp_path / "request_attempts.jsonl").read_text().splitlines()]
        assert starts[0]["sample_index"] == "0"
        assert starts[0]["status"] == "request_started"
        assert not (tmp_path / "usage.jsonl").exists()
    endpoint[0]["inspect_request"] = inspect
    model.inference([history()])
    starts = [json.loads(line) for line in (tmp_path / "request_attempts.jsonl").read_text().splitlines()]
    assert starts[0]["request_id"] == ledger(tmp_path)[0]["request_id"]
