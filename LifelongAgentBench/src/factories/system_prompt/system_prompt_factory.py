import json
from typing import Optional

from src.typings import Role, SystemPromptDict, ChatHistoryItem
from abc import ABC, abstractmethod


class SystemPromptFactoryInterface(ABC):
    @abstractmethod
    def construct(
        self, prompt_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        pass

    @abstractmethod
    def get_system_prompt_dict_deep_copy(self) -> SystemPromptDict:
        pass

    @abstractmethod
    def set(self, prompt_index: int, role: Role, content: str) -> None:
        pass


class SystemPromptFactory(SystemPromptFactoryInterface):
    def __init__(self, system_prompt_dict_path: str):
        super().__init__()
        self._system_prompt_dict = SystemPromptDict.model_validate(
            json.load(open(system_prompt_dict_path))
        )

    def construct(
        self, prompt_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        result = self._system_prompt_dict.value[str(prompt_index)]
        if expected_role is not None:
            assert result.role == expected_role
        return result

    def get_system_prompt_dict_deep_copy(self) -> SystemPromptDict:
        return self._system_prompt_dict.model_copy(deep=True)

    def set(self, prompt_index: int, role: Role, content: str) -> None:
        self._system_prompt_dict.set_system_prompt(prompt_index, role, content)
