from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import openai
import os
import json
import time
from pathlib import Path
from uuid import uuid4
from typing import Any, Optional, Sequence, Mapping, TypeGuard

from src.language_models.language_model import LanguageModel
from src.typings import (
    Role,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    ChatHistory,
    GenerationBudgetExceeded,
    LanguageModelUnknownException,
)


class OpenaiLanguageModel(LanguageModel):
    """
    To keep the name of the class consistent with the name of file, use OpenaiAgent instead of OpenAIAgent.
    """

    @staticmethod
    def _resolve_env_reference(
        value: Optional[str], parameter_name: str
    ) -> Optional[str]:
        if value is None or not value.startswith("env:"):
            return value
        environment_variable = value.removeprefix("env:")
        if not environment_variable:
            raise ValueError(f"Empty environment variable name for {parameter_name}.")
        resolved_value = os.environ.get(environment_variable)
        if resolved_value is None:
            raise ValueError(
                f"Environment variable {environment_variable!r} required by "
                f"{parameter_name} is not set."
            )
        return resolved_value

    def __init__(
        self,
        model_name: str,
        role_dict: Mapping[str, str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        maximum_prompt_token_count: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        trust_env: bool = True,
    ):
        """
        max_prompt_tokens: The maximum number of tokens that can be used in the prompt. It can be used to set the
            context limit manually. If it is set to None, the context limit will be the same as the context length of
            the model selected.
        """
        super().__init__(role_dict)
        resolved_model_name = self._resolve_env_reference(model_name, "model_name")
        assert resolved_model_name is not None
        self.model_name = resolved_model_name
        api_key = self._resolve_env_reference(api_key, "api_key")
        base_url = self._resolve_env_reference(base_url, "base_url")
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        # Hidden SDK retries cannot be audited or charged to a task budget.
        client_kwargs["max_retries"] = 0
        self.max_retries = 0 if max_retries is None else max_retries
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        if not trust_env:
            import httpx

            client_kwargs["http_client"] = httpx.Client(trust_env=False)
        self.client = OpenAI(**client_kwargs)
        self.maximum_prompt_token_count = maximum_prompt_token_count
        self.begin_sample("", None)
        self._audit_path: Path | None = None
        self._model_alias = ""
        self._run_id = ""

    @staticmethod
    def _is_valid_message_list(
        message_list: list[Mapping[str, str]],
    ) -> TypeGuard[list[ChatCompletionMessageParam]]:
        for message_dict in message_list:
            if (
                "role" not in message_dict.keys()
                or "content" not in message_dict.keys()
            ):
                return False
        return True

    def configure_audit(self, *, output_dir: str, model_alias: str, run_id: str) -> None:
        self._audit_path = Path(output_dir) / "usage.jsonl"
        self._model_alias = model_alias
        self._run_id = run_id

    def _record_audit(self, filename: str, entry: dict[str, Any]) -> None:
        entry.update(run_id=self._run_id, model_alias=self._model_alias,
                     model_name=self.model_name, sample_index=self.sample_index,
                     phase=self.usage_phase, timestamp=time.time())
        if self._audit_path is not None:
            path = self._audit_path.with_name(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _record_usage(self, entry: dict[str, Any]) -> None:
        # Both journals survive checkpoint rollback.
        self._record_audit("usage.jsonl", entry)

    def _get_completion_content(
        self, message_list: Sequence[ChatCompletionMessageParam],
        inference_config_dict: Mapping[str, Any],
    ) -> Sequence[str]:
        request = dict(inference_config_dict)
        if request.get("n", 1) != 1 or request.get("stream", False):
            self.audit_failure = "Only one non-streaming completion per request is supported."
            raise LanguageModelUnknownException(self.audit_failure)
        if self.maximum_prompt_token_count is not None:
            # Conservative byte-level guard, not a substitute for provider usage.
            # No post-charge prompt-limit failure is converted into training feedback.
            upper_bound = sum(len(str(message.get("content", "")).encode("utf-8"))
                              + 64 for message in message_list) + 256
            if upper_bound > self.maximum_prompt_token_count:
                self._record_usage(dict(request_id=uuid4().hex, attempt=0,
                                        status="prompt_guard", prompt_tokens=0,
                                        completion_tokens=0, total_tokens=0,
                                        missing_usage=False, latency_seconds=0))
                raise LanguageModelContextLimitException("Conservative prompt byte guard exceeded.")
        remaining = None if self.output_budget is None else self.output_budget - self.output_tokens_used
        if remaining is not None and remaining <= 0:
            self._record_usage(dict(request_id=uuid4().hex, attempt=0,
                                    status="budget_exhausted", prompt_tokens=0,
                                    completion_tokens=0, total_tokens=0,
                                    missing_usage=False, latency_seconds=0))
            raise GenerationBudgetExceeded("Per-task generation budget exhausted.")
        token_key = "max_completion_tokens" if "max_completion_tokens" in request else "max_tokens"
        if remaining is not None:
            configured_cap = request.get(token_key)
            request[token_key] = min(int(configured_cap), remaining) if configured_cap is not None else remaining
            if request[token_key] <= 0:
                raise ValueError("Generation cap must be positive.")
        request_id = uuid4().hex
        # Retry only explicit rate-limit rejections: a timeout/server error may have
        # generated billable tokens whose usage is unavailable.
        for attempt in range(1, self.max_retries + 2):
            self._record_audit("request_attempts.jsonl", dict(
                request_id=request_id, attempt=attempt, status="request_started"))
            start = time.monotonic()
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=message_list, **request)
            except Exception as error:
                context_error = isinstance(error, openai.BadRequestError) and (
                    "context length" in str(error).lower() or "context_length_exceeded" in str(error).lower())
                rate_limit = isinstance(error, openai.RateLimitError)
                self._record_usage(dict(request_id=request_id, attempt=attempt,
                                        status="context_limit" if context_error else "rate_limited" if rate_limit else "request_error",
                                        error_type=type(error).__name__,
                                        prompt_tokens=0 if context_error or rate_limit else None,
                                        completion_tokens=0 if context_error or rate_limit else None,
                                        total_tokens=0 if context_error or rate_limit else None,
                                        missing_usage=not (context_error or rate_limit),
                                        latency_seconds=time.monotonic()-start))
                if context_error:
                    raise LanguageModelContextLimitException("Provider context limit exceeded.") from error
                if rate_limit and attempt <= self.max_retries:
                    time.sleep(min(2 ** (attempt-1), 8))
                    continue
                self.audit_failure = f"Model request failed: {type(error).__name__}"
                raise LanguageModelUnknownException(self.audit_failure) from error
            usage = completion.usage
            def valid_count(value):
                return isinstance(value, int) and not isinstance(value, bool) and value >= 0
            known = usage is not None and valid_count(usage.completion_tokens) and valid_count(usage.prompt_tokens)
            if known and usage.total_tokens is not None:
                known = valid_count(usage.total_tokens) and usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
            output_tokens = int(usage.completion_tokens) if known else None
            input_tokens = int(usage.prompt_tokens) if known else None
            self._record_usage(dict(request_id=request_id, attempt=attempt, status="completed",
                                    prompt_tokens=input_tokens, completion_tokens=output_tokens,
                                    total_tokens=(int(usage.total_tokens) if usage.total_tokens is not None else input_tokens + output_tokens) if known else None,
                                    total_tokens_derived=bool(known and usage.total_tokens is None),
                                    missing_usage=not known, latency_seconds=time.monotonic()-start,
                                    finish_reasons=[choice.finish_reason for choice in completion.choices],
                                    requested_max_tokens=request.get(token_key)))
            if not known and (self.output_budget is not None or self._audit_path is not None):
                self.audit_failure = "Provider omitted token usage; shared generation budget cannot be verified."
                raise LanguageModelUnknownException(self.audit_failure)
            if known:
                if output_tokens < 0 or input_tokens < 0:
                    self.audit_failure = "Provider reported negative token usage."
                    raise LanguageModelUnknownException(self.audit_failure)
                self.output_tokens_used += output_tokens
                if (remaining is not None and output_tokens > remaining) or (request.get(token_key) is not None and output_tokens > int(request[token_key])):
                    self.audit_failure = "Provider exceeded the requested generation cap."
                    raise LanguageModelUnknownException(self.audit_failure)
                if self.maximum_prompt_token_count is not None and input_tokens > self.maximum_prompt_token_count:
                    self.audit_failure = "Provider prompt usage exceeded the configured input guard."
                    raise LanguageModelUnknownException(self.audit_failure)
            contents = [choice.message.content or "" for choice in completion.choices]
            if len(contents) != 1 or not contents[0].strip():
                self.audit_failure = "Model returned an empty or malformed completion."
                raise LanguageModelUnknownException(self.audit_failure)
            return contents
        raise RuntimeError("Unreachable retry state")

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        """
        system_prompt: It is usually called as system_prompt. But in OpenAI documents, it is called as developer_prompt.
            But in practice, using `message_list = [{"role": "developer", "content": self.system_prompt}]` will raise an
            error. So all after all, I call it as system_prompt.
            Reference:
            https://platform.openai.com/docs/guides/text-generation#messages-and-roles
            https://platform.openai.com/docs/api-reference/chat/create
        inference_config_dict: Other config for OpenAI().chat.completions.create.
            e.g.:
            max_completion_tokens: The maximum number of tokens that can be generated in the chat completion. Notice
                that max_tokens is deprecated.
            Reference:
            https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens
        """
        # region Construct batch_message_list
        message_list_prefix: list[ChatCompletionMessageParam]
        if len(system_prompt) > 0:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []
        batch_message_list: list[Sequence[ChatCompletionMessageParam]] = []
        for chat_history in batch_chat_history:
            conversion_result = self._convert_chat_history_to_message_list(chat_history)
            assert OpenaiLanguageModel._is_valid_message_list(conversion_result)
            batch_message_list.append(message_list_prefix + conversion_result)
        # endregion
        # region Generate output
        output_str_list: list[str] = []
        for message_list in batch_message_list:
            output_str_list.extend(
                self._get_completion_content(message_list, inference_config_dict)
            )
        # endregion
        # region Convert output to ChatHistoryItem
        return [
            ChatHistoryItem(role=Role.AGENT, content=output_str)
            for output_str in output_str_list
        ]
        # endregion
