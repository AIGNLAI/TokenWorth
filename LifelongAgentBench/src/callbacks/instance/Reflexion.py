import os

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import SessionEvaluationOutcome, SampleStatus
from src.utils import SafeLogger
from .lifelong_memory_utils import (
    ReflectionRecord,
    dump_json_state_list,
    extract_question_from_session,
    inject_memory_block,
    load_json_state_list,
    parse_bullet_list,
    rank_texts_by_relevance,
    run_single_prompt,
    truncate_text,
    truncate_text_list_total,
)


class ReflexionCallback(Callback):
    def __init__(
        self,
        language_model: LanguageModel,
        top_k: int,
        max_trajectory_length: int = 2000,
        max_memory_count: int = 256,
        max_injected_chars_total: int = 2000,
        use_evaluator_outcome_only: bool = False,
        auxiliary_max_tokens: int | None = None,
    ):
        super().__init__()
        assert top_k > 0
        assert max_memory_count > 0
        assert max_injected_chars_total > 0
        assert auxiliary_max_tokens is None or auxiliary_max_tokens > 0
        self.language_model = language_model
        self.top_k = top_k
        self.max_trajectory_length = max_trajectory_length
        self.max_memory_count = max_memory_count
        self.max_injected_chars_total = max_injected_chars_total
        self.use_evaluator_outcome_only = use_evaluator_outcome_only
        self.auxiliary_max_tokens = auxiliary_max_tokens
        self.reflection_record_list: list[ReflectionRecord] = []

    def _get_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "reflection_record_list.json")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def restore_state(self) -> None:
        self.reflection_record_list = [
            ReflectionRecord.model_validate(item)
            for item in load_json_state_list(self._get_state_path(), ReflectionRecord)
        ]

    def _build_reflection_prompt(self, callback_args: CallbackArguments) -> str:
        session = callback_args.current_session
        query = extract_question_from_session(session)
        role_dict = callback_args.session_context.agent.get_role_dict()
        if session.chat_history.get_value_length() > 3:
            trajectory = session.chat_history.get_value_str(
                role_dict,
                start_index=3,
                end_index=None,
            )
        else:
            trajectory = ""
        return f"""You are writing a short self-reflection for a lifelong LLM agent.

Task:
{query}

Outcome:
The attempt was not successful.

Trajectory:
{truncate_text(trajectory, self.max_trajectory_length)}

Write 1 to 3 concise bullet points. Each bullet should be a reusable lesson or caution that could help on similar future tasks.
Avoid mentioning sample ids or overly task-specific details.
"""

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            len(self.reflection_record_list) == 0
            or session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            return
        current_query = session.chat_history.get_item_deep_copy(-1).content
        selected_index_list = rank_texts_by_relevance(
            current_query,
            [record.search_text for record in self.reflection_record_list],
            self.top_k,
        )
        if len(selected_index_list) == 0:
            return
        inject_memory_block(
            session.chat_history,
            "Relevant lessons from past failures",
            truncate_text_list_total(
                [
                    self.reflection_record_list[index].reflection
                    for index in selected_index_list
                ],
                self.max_injected_chars_total,
            ),
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        is_correct = (
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            and (
                self.use_evaluator_outcome_only
                or session.sample_status == SampleStatus.COMPLETED
            )
        )
        if is_correct:
            return
        try:
            reflection_text = run_single_prompt(
                self.language_model,
                self._build_reflection_prompt(callback_args),
                max_tokens=self.auxiliary_max_tokens,
            )
        except Exception as error:
            SafeLogger.warning(
                f"[ReflexionCallback] Failed to generate reflection: {error}"
            )
            return
        query = extract_question_from_session(session)
        reflection_list = parse_bullet_list(reflection_text, max_item_count=3)
        if len(reflection_list) == 0:
            return
        for reflection in reflection_list:
            search_text = f"{query}\n{reflection}"
            self.reflection_record_list.append(
                ReflectionRecord(
                    query=query,
                    reflection=reflection,
                    search_text=search_text,
                )
            )
        if len(self.reflection_record_list) > self.max_memory_count:
            self.reflection_record_list = self.reflection_record_list[
                -self.max_memory_count :
            ]

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_state_path(), self.reflection_record_list)
