import os
from typing import Mapping

from src.agents import Agent
from src.language_models import LanguageModel
from src.tasks import DatasetItem, Task
from src.typings import AssignmentConfig

from .callback import Callback
from .instance import (
    ConsecutiveAbnormalAgentInferenceProcessHandlingCallback,
    CurrentSessionSavingCallback,
    EvoSCCallback,
    EvoSkillCallback,
    Mem0Callback,
    MemPCallback,
    MemRLCallback,
    MemSkillCallback,
    ProcessMemoryMonitorCallback,
    RAGExperienceRetrievalCallback,
    ReflexionCallback,
    TokenWorthMemoryCallback,
    TokenWorthOSIdiomMemoryCallback,
)


CALLBACK_CLASS_DICT: dict[str, type[Callback]] = {
    callback_class.__name__: callback_class
    for callback_class in (
        ConsecutiveAbnormalAgentInferenceProcessHandlingCallback,
        CurrentSessionSavingCallback,
        EvoSCCallback,
        EvoSkillCallback,
        Mem0Callback,
        MemPCallback,
        MemRLCallback,
        MemSkillCallback,
        ProcessMemoryMonitorCallback,
        RAGExperienceRetrievalCallback,
        ReflexionCallback,
        TokenWorthMemoryCallback,
        TokenWorthOSIdiomMemoryCallback,
    )
}

LANGUAGE_MODEL_CALLBACK_NAME_SET = {
    EvoSCCallback.__name__,
    EvoSkillCallback.__name__,
    Mem0Callback.__name__,
    MemPCallback.__name__,
    MemSkillCallback.__name__,
    ReflexionCallback.__name__,
}


class CallbackConstructor:
    @staticmethod
    def construct(
        assignment_config: AssignmentConfig,
        task: Task[DatasetItem],
        agent: Agent,
        language_model_dict: Mapping[str, LanguageModel],
    ) -> dict[str, Callback]:
        del task, agent
        instanced_unique_callback_name_set: set[str] = set()

        for general_instance_factory in assignment_config.callback_dict.values():
            target_class_name = general_instance_factory.module.split(".")[-1]
            callback_class = CALLBACK_CLASS_DICT.get(target_class_name)
            if callback_class is None:
                raise NotImplementedError(
                    f"Callback {target_class_name} is not available in this release."
                )

            if target_class_name == CurrentSessionSavingCallback.__name__:
                general_instance_factory.parameters["saving_path"] = os.path.join(
                    assignment_config.output_dir, "current_session.json"
                )
            elif target_class_name in LANGUAGE_MODEL_CALLBACK_NAME_SET:
                language_model_name = general_instance_factory.parameters[
                    "language_model"
                ]
                general_instance_factory.parameters["language_model"] = (
                    language_model_dict[language_model_name]
                )

            if callback_class.is_unique():
                assert target_class_name not in instanced_unique_callback_name_set
                instanced_unique_callback_name_set.add(target_class_name)

        callback_dict: dict[str, Callback] = {
            callback_id: general_instance_factory.create()
            for callback_id, general_instance_factory in assignment_config.callback_dict.items()
        }
        for callback_id, callback in callback_dict.items():
            callback.set_state_dir(
                os.path.join(
                    assignment_config.output_dir, "callback_state", callback_id
                )
            )
        return callback_dict
