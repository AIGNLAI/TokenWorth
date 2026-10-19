import os

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import SampleStatus, SessionEvaluationOutcome
from src.utils import SafeLogger
from .lifelong_memory_utils import (
    ExperienceRecord,
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

class EvoSCCallback(Callback):
    def __init__(
        self,
        language_model: LanguageModel,
        top_k_success: int = 4,
        top_k_failure: int = 4,
        max_trajectory_length: int = 2000,
        max_success_count: int = 64,
        max_failure_count: int = 64,
        max_summary_count: int = 32,
        max_injected_chars_total: int = 2000,
        use_evaluator_outcome_only: bool = False,
        auxiliary_max_tokens: int | None = None,
        max_retrieved_items: int | None = None,
    ):
        super().__init__()
        assert top_k_success > 0
        assert top_k_failure > 0
        assert max_success_count > 0
        assert max_failure_count > 0
        assert max_summary_count > 0
        assert max_injected_chars_total > 0
        assert auxiliary_max_tokens is None or auxiliary_max_tokens > 0
        assert max_retrieved_items is None or max_retrieved_items > 0
        self.max_retrieved_items = max_retrieved_items
        self.language_model = language_model
        self.top_k_success = top_k_success
        self.top_k_failure = top_k_failure
        self.max_trajectory_length = max_trajectory_length
        self.max_success_count = max_success_count
        self.max_failure_count = max_failure_count
        self.max_summary_count = max_summary_count
        self.max_injected_chars_total = max_injected_chars_total
        self.use_evaluator_outcome_only = use_evaluator_outcome_only
        self.auxiliary_max_tokens = auxiliary_max_tokens
        self.success_record_list: list[ExperienceRecord] = []
        self.failure_reflection_record_list: list[ReflectionRecord] = []
        self.consolidated_summary_list: list[str] = []
        self.consolidated_summary_dict: dict[str, list[str]] = {}

    def _get_memory_bucket(self, text: str) -> str:
        return "general"

    def _build_search_text(self, query: str, text: str) -> str:
        return f"{query}\n{text}"

    def _select_record_index_list(
        self,
        current_query: str,
        search_text_list: list[str],
        top_k: int,
    ) -> list[int]:
        if len(search_text_list) == 0:
            return []
        return rank_texts_by_relevance(
            current_query,
            search_text_list,
            min(top_k, len(search_text_list)),
        )

    def _merge_unique_bullets(
        self, text_list: list[str], max_item_count: int
    ) -> list[str]:
        result: list[str] = []
        normalized_text_set: set[str] = set()
        for text in text_list:
            normalized_text = " ".join(text.lower().split())
            if len(normalized_text) == 0 or normalized_text in normalized_text_set:
                continue
            normalized_text_set.add(normalized_text)
            result.append(text)
            if len(result) >= max_item_count:
                break
        return result

    def _build_success_strategy_prompt(self, callback_args: CallbackArguments) -> str:
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
        return f"""You are extracting short reusable success strategies for EvoSC.

Task:
{query}

Successful trajectory:
{truncate_text(trajectory, self.max_trajectory_length)}

Write 1 to 3 concise bullet points describing reusable tactics that helped solve this task.
Focus on stable procedures, checks, or decision rules.
Avoid copying long trajectory details or sample-specific wording.
"""

    def _get_success_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "evosc_success_record_list.json")

    def _get_failure_state_path(self) -> str:
        return os.path.join(
            self.get_state_dir(), "evosc_failure_reflection_record_list.json"
        )

    def _get_summary_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "evosc_consolidated_summary.json")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def restore_state(self) -> None:
        self.success_record_list = [
            ExperienceRecord.model_validate(item)
            for item in load_json_state_list(
                self._get_success_state_path(), ExperienceRecord
            )
        ]
        self.failure_reflection_record_list = [
            ReflectionRecord.model_validate(item)
            for item in load_json_state_list(
                self._get_failure_state_path(), ReflectionRecord
            )
        ]
        if os.path.exists(self._get_summary_state_path()):
            summary_record_list = load_json_state_list(
                self._get_summary_state_path(), ReflectionRecord
            )
            self.consolidated_summary_list = []
            self.consolidated_summary_dict = {}
            for record in summary_record_list:
                bucket = record.query
                if bucket == "consolidated":
                    self.consolidated_summary_list.append(record.reflection)
                    bucket = "general"
                self.consolidated_summary_dict.setdefault(bucket, []).append(
                    record.reflection
                )

    def _build_failure_reflection_prompt(self, callback_args: CallbackArguments) -> str:
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
        return f"""You are extracting error-prone patterns for EvoSC.

Task:
{query}

Failed trajectory:
{truncate_text(trajectory, self.max_trajectory_length)}

Write 1 to 3 concise bullet points describing reusable failure patterns or corrective insights.
Avoid sample-specific details.
"""

    def _build_consolidation_prompt(
        self,
        bucket: str,
        query: str,
        latest_success_snippet_list: list[str],
        latest_failure_reflection_list: list[str],
    ) -> str:
        current_summary = "\n".join(
            f"- {item}" for item in self.consolidated_summary_dict.get(bucket, [])
        )
        success_block = "\n".join(f"- {item}" for item in latest_success_snippet_list)
        failure_block = "\n".join(
            f"- {item}" for item in latest_failure_reflection_list
        )
        return f"""You are consolidating long-term intuition for an LLM agent.

Current consolidated intuition:
{current_summary if current_summary else "- None"}

Latest successful experience cues:
{success_block if success_block else "- None"}

Latest failure-derived cues:
{failure_block if failure_block else "- None"}

Update the consolidated intuition for future {bucket} tasks similar to:
{query}

Write at most 5 concise bullet points.
Prefer high-precision guidance over broad generic advice.
"""

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            return
        current_query = session.chat_history.get_item_deep_copy(-1).content
        current_bucket = self._get_memory_bucket(current_query)
        if self.max_retrieved_items is not None:
            # Recall each experience type, then apply the shared injection-item cap.
            # A summary bucket is one compact controller record.
            summary_list = list(self.consolidated_summary_dict.get(current_bucket, []))
            if current_bucket != "general":
                summary_list += self.consolidated_summary_dict.get("general", [])
            candidates = [(item, item) for item in summary_list]
            candidates += [(record.search_text, record.snippet) for record in self.success_record_list]
            candidates += [(record.search_text, record.reflection) for record in self.failure_reflection_record_list]
            indices = rank_texts_by_relevance(current_query, [item[0] for item in candidates], self.max_retrieved_items)
            injected = truncate_text_list_total([candidates[index][1] for index in indices], self.max_injected_chars_total)
            inject_memory_block(session.chat_history, "EvoSC retrieved experience", injected)
            return
        remaining_chars = self.max_injected_chars_total
        summary_list = self.consolidated_summary_dict.get(current_bucket, [])
        if current_bucket != "general":
            summary_list += self.consolidated_summary_dict.get("general", [])
        if len(summary_list) > 0 and remaining_chars > 0:
            selected_summary_index_list = rank_texts_by_relevance(
                current_query,
                summary_list,
                min(2, len(summary_list)),
            )
            injected_summary_list = truncate_text_list_total(
                [summary_list[index] for index in selected_summary_index_list],
                remaining_chars,
            )
            remaining_chars -= sum(len(item) for item in injected_summary_list)
            inject_memory_block(
                session.chat_history,
                "EvoSC consolidated intuition",
                injected_summary_list,
            )
        if len(self.success_record_list) > 0 and remaining_chars > 0:
            selected_success_index_list = self._select_record_index_list(
                current_query,
                [record.search_text for record in self.success_record_list],
                min(2, self.top_k_success),
            )
            if len(selected_success_index_list) > 0:
                injected_success_list = truncate_text_list_total(
                    [
                        self.success_record_list[index].snippet
                        for index in selected_success_index_list
                    ],
                    remaining_chars,
                )
                remaining_chars -= sum(len(item) for item in injected_success_list)
                inject_memory_block(
                    session.chat_history,
                    "EvoSC successful strategies",
                    injected_success_list,
                )
        if len(self.failure_reflection_record_list) > 0 and remaining_chars > 0:
            selected_failure_index_list = self._select_record_index_list(
                current_query,
                [record.search_text for record in self.failure_reflection_record_list],
                min(2, self.top_k_failure),
            )
            if len(selected_failure_index_list) > 0:
                inject_memory_block(
                    session.chat_history,
                    "EvoSC cautions from past failures",
                    truncate_text_list_total(
                        [
                            self.failure_reflection_record_list[index].reflection
                            for index in selected_failure_index_list
                        ],
                        remaining_chars,
                    ),
                )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            not self.use_evaluator_outcome_only
            and session.sample_status != SampleStatus.COMPLETED
        ):
            return
        query = extract_question_from_session(session)
        bucket = self._get_memory_bucket(query)
        latest_success_snippet_list: list[str] = []
        latest_failure_reflection_list: list[str] = []
        if session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT:
            try:
                success_strategy_text = run_single_prompt(
                    self.language_model,
                    self._build_success_strategy_prompt(callback_args),
                    max_tokens=self.auxiliary_max_tokens,
                )
            except Exception as error:
                SafeLogger.warning(
                    f"[EvoSCCallback] Failed to generate success strategies: {error}"
                )
                success_strategy_text = ""
            success_strategy_list = self._merge_unique_bullets(
                parse_bullet_list(success_strategy_text, max_item_count=3),
                max_item_count=3,
            )
            if len(success_strategy_list) > 0:
                snippet_text = "\n".join(
                    f"Successful strategy: {strategy}"
                    for strategy in success_strategy_list
                )
                snippet_record = ExperienceRecord(
                    query=query,
                    snippet=snippet_text,
                    search_text=self._build_search_text(query, snippet_text),
                    outcome=session.evaluation_record.outcome,
                )
                self.success_record_list.append(snippet_record)
                if len(self.success_record_list) > self.max_success_count:
                    self.success_record_list = self.success_record_list[
                        -self.max_success_count :
                    ]
                latest_success_snippet_list.extend(success_strategy_list)
        else:
            try:
                reflection_text = run_single_prompt(
                    self.language_model,
                    self._build_failure_reflection_prompt(callback_args),
                    max_tokens=self.auxiliary_max_tokens,
                )
            except Exception as error:
                SafeLogger.warning(
                    f"[EvoSCCallback] Failed to generate failure reflections: {error}"
                )
                return
            for reflection in self._merge_unique_bullets(
                parse_bullet_list(reflection_text, max_item_count=3),
                max_item_count=3,
            ):
                self.failure_reflection_record_list.append(
                    ReflectionRecord(
                        query=query,
                        reflection=reflection,
                        search_text=self._build_search_text(query, reflection),
                    )
                )
                if len(self.failure_reflection_record_list) > self.max_failure_count:
                    self.failure_reflection_record_list = (
                        self.failure_reflection_record_list[-self.max_failure_count :]
                    )
                latest_failure_reflection_list.append(reflection)
        try:
            consolidation_text = run_single_prompt(
                self.language_model,
                self._build_consolidation_prompt(
                    bucket,
                    query,
                    latest_success_snippet_list,
                    latest_failure_reflection_list,
                ),
                max_tokens=self.auxiliary_max_tokens,
            )
        except Exception as error:
            SafeLogger.warning(
                f"[EvoSCCallback] Failed to update consolidated intuition: {error}"
            )
            return
        self.consolidated_summary_dict[bucket] = self._merge_unique_bullets(
            parse_bullet_list(
                consolidation_text,
                max_item_count=5,
            ),
            max_item_count=5,
        )
        if len(self.consolidated_summary_dict) > self.max_summary_count:
            retained_bucket_list = list(self.consolidated_summary_dict)[
                -self.max_summary_count :
            ]
            self.consolidated_summary_dict = {
                retained_bucket: self.consolidated_summary_dict[retained_bucket]
                for retained_bucket in retained_bucket_list
            }
        self._prune_summary_items()

    def _prune_summary_items(self) -> None:
        remaining = self.max_summary_count
        retained = {}
        for bucket, items in reversed(list(self.consolidated_summary_dict.items())):
            if remaining <= 0:
                break
            chosen = items[-remaining:]
            if chosen:
                retained[bucket] = chosen
                remaining -= len(chosen)
        self.consolidated_summary_dict = dict(reversed(list(retained.items())))
        self.consolidated_summary_list = self.consolidated_summary_dict.get("general", [])

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        self._prune_summary_items()
        dump_json_state_list(self._get_success_state_path(), self.success_record_list)
        dump_json_state_list(
            self._get_failure_state_path(),
            self.failure_reflection_record_list,
        )
        dump_json_state_list(
            self._get_summary_state_path(),
            [
                ReflectionRecord(
                    query=bucket,
                    reflection=item,
                    search_text=item,
                )
                for bucket, item_list in self.consolidated_summary_dict.items()
                for item in item_list
            ],
        )
