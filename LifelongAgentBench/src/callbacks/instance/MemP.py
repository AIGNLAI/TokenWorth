import os
import re

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import SampleStatus, SessionEvaluationOutcome
from src.utils import SafeLogger
from .advanced_memory_utils import (
    MemPRecord,
    ProcedureMemoryKind,
    ProcedureMemoryStatus,
    build_trajectory_text,
    coerce_str_list,
    extract_json_object_from_text,
    merge_unique_texts,
    rank_memp_records,
    render_record_list,
)
from .lifelong_memory_utils import (
    dump_json_state_list,
    extract_question_from_session,
    inject_memory_block,
    load_json_state_list,
    normalize_memory_text,
    run_single_prompt,
    truncate_text_list_total,
)

class MemPCallback(Callback):
    def __init__(
        self,
        language_model: LanguageModel,
        top_k: int,
        max_memory_count: int = 256,
        max_trajectory_length: int = 2000,
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
        self.max_memory_count = max_memory_count
        self.max_trajectory_length = max_trajectory_length
        self.max_injected_chars_total = max_injected_chars_total
        self.use_evaluator_outcome_only = use_evaluator_outcome_only
        self.auxiliary_max_tokens = auxiliary_max_tokens
        self.procedure_record_list: list[MemPRecord] = []
        self.current_step = 0
        self.current_retrieved_record_index_list: list[int] = []

    def _normalize_memory_reference(self, memory_text: str) -> str:
        normalized_memory_text = memory_text.strip()
        match = re.match(
            r"^\[(detailed|abstract|caution)\]\s*(.*)$",
            normalized_memory_text,
        )
        if match is not None:
            normalized_memory_text = match.group(2).strip()
        return normalized_memory_text

    def _is_reusable_memory(self, memory_text: str) -> bool:
        if len(memory_text) > 180:
            return False
        if "```" in memory_text or "\n" in memory_text:
            return False
        return True

    def _select_retrieval_index_list(self, query: str) -> list[int]:
        selected_index_list: list[int] = []
        selected_index_set: set[int] = set()
        per_kind_limit_dict = {
            ProcedureMemoryKind.ABSTRACT: 1,
            ProcedureMemoryKind.DETAILED: min(2, self.top_k),
            ProcedureMemoryKind.CAUTION: 1,
        }
        for kind, limit in per_kind_limit_dict.items():
            for index in rank_memp_records(
                query,
                self.procedure_record_list,
                limit,
                self.current_step,
                allowed_kind_set={kind},
            ):
                if index in selected_index_set:
                    continue
                selected_index_list.append(index)
                selected_index_set.add(index)
        if len(selected_index_list) < self.top_k:
            for index in rank_memp_records(
                query,
                self.procedure_record_list,
                self.top_k,
                self.current_step,
            ):
                if index in selected_index_set:
                    continue
                selected_index_list.append(index)
                selected_index_set.add(index)
                if len(selected_index_list) >= self.top_k:
                    break
        return selected_index_list[: min(self.top_k, 4)]

    def _get_memory_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memp_record_list.json")

    def _get_step_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memp_step.txt")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def restore_state(self) -> None:
        self.procedure_record_list = [
            MemPRecord.model_validate(item)
            for item in load_json_state_list(self._get_memory_state_path(), MemPRecord)
        ]
        if os.path.exists(self._get_step_state_path()):
            self.current_step = int(
                open(self._get_step_state_path(), "r").read().strip()
            )

    def _build_success_prompt(self, callback_args: CallbackArguments) -> str:
        session = callback_args.current_session
        query = extract_question_from_session(session)
        trajectory = build_trajectory_text(
            session,
            callback_args.session_context.agent.get_role_dict(),
            self.max_trajectory_length,
        )
        return f"""You are building MemP, a procedural memory repository for an LLM agent.

Task:
{query}

Trajectory:
{trajectory}

Distill this successful experience into:
1. fine-grained procedural instructions
2. higher-level reusable script-like abstractions

Rules:
- Return at most 3 detailed_steps and at most 2 abstractions.
- Each item must be a single-line reusable procedure under 160 characters.
- Write task-agnostic procedure patterns; avoid task-specific names, literal values, and sample outputs.
- Do not include complete executable answers or code blocks.

Return JSON only in a fenced code block:
```json
{{
  "detailed_steps": ["step-by-step procedural memory"],
  "abstractions": ["high-level reusable script"]
}}
```
"""

    def _build_failure_prompt(self, callback_args: CallbackArguments) -> str:
        session = callback_args.current_session
        query = extract_question_from_session(session)
        trajectory = build_trajectory_text(
            session,
            callback_args.session_context.agent.get_role_dict(),
            self.max_trajectory_length,
        )
        retrieved_memory_list = [
            self.procedure_record_list[index].get_render_text()
            for index in self.current_retrieved_record_index_list
        ]
        retrieved_memory_block = "\n".join(f"- {text}" for text in retrieved_memory_list)
        return f"""You are updating MemP after an unsuccessful trajectory.

Task:
{query}

Retrieved procedural memories:
{retrieved_memory_block if retrieved_memory_block else "- None"}

Failed trajectory:
{trajectory}

Produce:
1. cautions or corrections that should be stored for similar future tasks
2. outdated retrieved memories that should be deprecated if they were misleading

Rules:
- Return at most 3 cautions and at most 2 deprecate items.
- Each item must be a single-line reusable statement under 160 characters.
- For deprecate, copy the exact retrieved memory text without the [kind] prefix when possible.
- Write task-agnostic cautions; avoid task-specific names, literal values, and sample outputs.
- Do not include complete executable answers or code blocks.

Return JSON only in a fenced code block:
```json
{{
  "cautions": ["new caution or correction"],
  "deprecate": ["outdated or misleading memory text"]
}}
```
"""

    def _upsert_memory(
        self,
        memory_text: str,
        kind: ProcedureMemoryKind,
        query: str,
    ) -> None:
        normalized_memory_text = self._normalize_memory_reference(memory_text)
        if not self._is_reusable_memory(normalized_memory_text):
            return
        normalized_memory = normalize_memory_text(normalized_memory_text)
        if len(normalized_memory) == 0:
            return
        for record in self.procedure_record_list:
            if normalize_memory_text(record.memory) != normalized_memory:
                continue
            record.status = ProcedureMemoryStatus.ACTIVE
            record.support_count += 1
            record.last_access_step = self.current_step
            if len(query) > len(record.source_query):
                record.source_query = query
            return
        self.procedure_record_list.append(
            MemPRecord(
                memory=normalized_memory_text,
                kind=kind,
                status=ProcedureMemoryStatus.ACTIVE,
                source_query=query,
                created_step=self.current_step,
                last_access_step=self.current_step,
                support_count=1,
                contradiction_count=0,
            )
        )

    def _deprecate_memory(self, memory_text: str) -> None:
        normalized_memory = normalize_memory_text(
            self._normalize_memory_reference(memory_text)
        )
        if len(normalized_memory) == 0:
            return
        for record in self.procedure_record_list:
            if normalize_memory_text(record.memory) != normalized_memory:
                continue
            record.contradiction_count += 1
            record.last_access_step = self.current_step
            if record.contradiction_count > record.support_count:
                record.status = ProcedureMemoryStatus.DEPRECATED

    def _prune_memory(self) -> None:
        if len(self.procedure_record_list) <= self.max_memory_count:
            return
        retained_index_list = rank_memp_records(
            "",
            self.procedure_record_list,
            self.max_memory_count,
            self.current_step,
        )
        retained_index_set = set(retained_index_list)
        self.procedure_record_list = [
            record
            for index, record in enumerate(self.procedure_record_list)
            if index in retained_index_set
            or record.status == ProcedureMemoryStatus.ACTIVE
        ][: self.max_memory_count]

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            len(self.procedure_record_list) == 0
            or session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            self.current_retrieved_record_index_list = []
            return
        current_query = session.chat_history.get_item_deep_copy(-1).content
        selected_index_list = self._select_retrieval_index_list(current_query)
        for index in selected_index_list:
            self.procedure_record_list[index].last_access_step = self.current_step
        self.current_retrieved_record_index_list = selected_index_list
        inject_memory_block(
            callback_args.current_session.chat_history,
            "MemP procedural memories (advisory; do not copy old identifiers)",
            truncate_text_list_total(
                render_record_list(
                    [self.procedure_record_list[index] for index in selected_index_list]
                ),
                self.max_injected_chars_total,
            ),
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        self.current_step += 1
        session = callback_args.current_session
        if (
            not self.use_evaluator_outcome_only
            and session.sample_status != SampleStatus.COMPLETED
        ):
            return
        query = extract_question_from_session(session)
        if session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT:
            try:
                response_text = run_single_prompt(
                    self.language_model,
                    self._build_success_prompt(callback_args),
                    max_tokens=self.auxiliary_max_tokens,
                )
                response_dict = extract_json_object_from_text(
                    response_text,
                    required_key_list=["detailed_steps", "abstractions"],
                )
            except Exception as error:
                SafeLogger.warning(f"[MemPCallback] Failed to build memories: {error}")
                return
            for detail in merge_unique_texts(
                coerce_str_list(response_dict.get("detailed_steps")),
                max_count=5,
            ):
                self._upsert_memory(detail, ProcedureMemoryKind.DETAILED, query)
            for abstraction in merge_unique_texts(
                coerce_str_list(response_dict.get("abstractions")),
                max_count=5,
            ):
                self._upsert_memory(abstraction, ProcedureMemoryKind.ABSTRACT, query)
        else:
            try:
                response_text = run_single_prompt(
                    self.language_model,
                    self._build_failure_prompt(callback_args),
                    max_tokens=self.auxiliary_max_tokens,
                )
                response_dict = extract_json_object_from_text(
                    response_text,
                    required_key_list=["cautions", "deprecate"],
                )
            except Exception as error:
                SafeLogger.warning(f"[MemPCallback] Failed to update memories: {error}")
                return
            for caution in merge_unique_texts(
                coerce_str_list(response_dict.get("cautions")),
                max_count=5,
            ):
                self._upsert_memory(caution, ProcedureMemoryKind.CAUTION, query)
            for deprecated_memory in merge_unique_texts(
                coerce_str_list(response_dict.get("deprecate")),
                max_count=5,
            ):
                self._deprecate_memory(deprecated_memory)
        self._prune_memory()

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_memory_state_path(), self.procedure_record_list)
        open(self._get_step_state_path(), "w").write(str(self.current_step))
