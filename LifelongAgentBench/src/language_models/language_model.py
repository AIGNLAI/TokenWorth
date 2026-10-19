from abc import ABC, abstractmethod
from typing import Sequence, Mapping, Any, Optional, final

from src.typings import (
    ChatHistory,
    ChatHistoryItem,
    Role,
    ModelException,
    LanguageModelUnknownException,
)


class LanguageModel(ABC):
    def __init__(self, role_dict: Mapping[str, str]) -> None:
        self.role_dict: Mapping[Role, str] = {
            Role(role): role_dict[role] for role in Role
        }

    def configure_audit(self, *, output_dir: str, model_alias: str, run_id: str) -> None:
        """Adapters can persist request usage without conversation or credentials."""

    def begin_sample(self, sample_index: str, output_budget: int | None) -> None:
        self.sample_index = sample_index
        self.output_budget = output_budget
        self.output_tokens_used = 0
        self.audit_failure: str | None = None
        self.usage_phase = "agent"

    def set_usage_context(self, *, sample_index: str | None = None,
                          phase: str = "agent", output_budget: int | None = None) -> None:
        if phase not in {"agent", "auxiliary"}:
            raise ValueError("Unknown model usage phase.")
        self.usage_phase = phase
        if sample_index is not None:
            self.sample_index = sample_index

    def _convert_chat_history_to_message_list(
        self, chat_history: ChatHistory
    ) -> list[Mapping[str, str]]:
        message_list: list[Mapping[str, str]] = []
        for item_index in range(chat_history.get_value_length()):
            message = chat_history.get_item_deep_copy(item_index)
            message_list.append(
                {
                    "role": self.role_dict[message.role],
                    "content": message.content,
                }
            )
        return message_list

    @final
    def inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Sequence[ChatHistoryItem]:
        for chat_history in batch_chat_history:
            assert chat_history.get_item_deep_copy(-1).role == Role.USER
        try:
            if inference_config_dict is None:
                inference_config_dict = {}
            inference_result = self._inference(
                batch_chat_history, inference_config_dict, system_prompt
            )
        except ModelException as e:
            if isinstance(e, LanguageModelUnknownException):
                self.audit_failure = f"Model adapter failure: {type(e).__name__}"
            raise e
        except Exception as e:
            self.audit_failure = f"Unexpected model adapter failure: {type(e).__name__}"
            raise LanguageModelUnknownException(str(e)) from e
        return inference_result

    @abstractmethod
    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        pass
