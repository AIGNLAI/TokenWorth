import os

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import SampleStatus, SessionEvaluationOutcome
from src.utils import SafeLogger
from .advanced_memory_utils import (
    Mem0Record,
    build_trajectory_text,
    coerce_str_list,
    extract_json_object_from_text,
    merge_unique_texts,
    normalize_entities,
    rank_mem0_records,
    render_mem0_record,
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


class Mem0Callback(Callback):
    def __init__(
        self,
        language_model: LanguageModel,
        top_k: int,
        max_memory_count: int = 512,
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
        self.memory_record_list: list[Mem0Record] = []
        self.current_step = 0

    def _get_memory_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "mem0_record_list.json")

    def _get_step_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "mem0_step.txt")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def restore_state(self) -> None:
        self.memory_record_list = [
            Mem0Record.model_validate(item)
            for item in load_json_state_list(self._get_memory_state_path(), Mem0Record)
        ]
        if os.path.exists(self._get_step_state_path()):
            self.current_step = int(
                open(self._get_step_state_path(), "r").read().strip()
            )

    def _build_memory_extraction_prompt(self, callback_args: CallbackArguments) -> str:
        session = callback_args.current_session
        query = extract_question_from_session(session)
        trajectory = build_trajectory_text(
            session,
            callback_args.session_context.agent.get_role_dict(),
            self.max_trajectory_length,
        )
        return f"""You are Mem0, a long-term memory extractor for an LLM agent.

Task:
{query}

Outcome:
{session.evaluation_record.outcome.value}

Trajectory:
{trajectory}

Extract at most 5 durable memories that should remain useful in future related tasks.
Prioritize stable facts, constraints, preferences, and reusable procedures.
Avoid sample-specific wording and avoid duplicate memories.

Return JSON only in a fenced code block:
```json
{{
  "memories": [
    {{
      "memory": "atomic reusable memory",
      "category": "fact|constraint|preference|procedure",
      "entities": ["entity1", "entity2"]
    }}
  ]
}}
```
"""

    def _upsert_memory(
        self,
        memory_text: str,
        category: str,
        entity_list: list[str],
        query: str,
        outcome: SessionEvaluationOutcome,
    ) -> None:
        normalized_memory = normalize_memory_text(memory_text)
        if len(normalized_memory) == 0:
            return
        for record in self.memory_record_list:
            if normalize_memory_text(record.memory) != normalized_memory:
                continue
            record.mentions += 1
            record.last_access_step = self.current_step
            record.importance += 0.4 if outcome == SessionEvaluationOutcome.CORRECT else 0.2
            record.entities = normalize_entities(record.entities + entity_list)
            if len(query) > len(record.source_query):
                record.source_query = query
            if outcome == SessionEvaluationOutcome.CORRECT:
                record.outcome = outcome
            return
        self.memory_record_list.append(
            Mem0Record(
                memory=memory_text,
                category=category,
                entities=normalize_entities(entity_list),
                source_query=query,
                outcome=outcome,
                created_step=self.current_step,
                last_access_step=self.current_step,
                mentions=1,
                importance=1.0 if outcome == SessionEvaluationOutcome.CORRECT else 0.7,
            )
        )

    def _prune_memory(self) -> None:
        if len(self.memory_record_list) <= self.max_memory_count:
            return
        retained_index_list = rank_mem0_records(
            "",
            self.memory_record_list,
            self.max_memory_count,
            self.current_step,
        )
        self.memory_record_list = [
            self.memory_record_list[index] for index in retained_index_list
        ]

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            len(self.memory_record_list) == 0
            or session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            return
        current_query = session.chat_history.get_item_deep_copy(-1).content
        eligible_record_list = [
            record
            for record in self.memory_record_list
            if record.outcome == SessionEvaluationOutcome.CORRECT
        ]
        if len(eligible_record_list) == 0:
            eligible_record_list = self.memory_record_list
        selected_index_list = rank_mem0_records(
            current_query,
            eligible_record_list,
            self.top_k,
            self.current_step,
        )
        selected_record_list = [eligible_record_list[index] for index in selected_index_list]
        for record in selected_record_list:
            record.last_access_step = self.current_step
            record.mentions += 1
            record.importance += 0.05
        inject_memory_block(
            session.chat_history,
            "Mem0 long-term memories",
            truncate_text_list_total(
                [render_mem0_record(record) for record in selected_record_list],
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
        try:
            response_text = run_single_prompt(
                self.language_model,
                self._build_memory_extraction_prompt(callback_args),
                max_tokens=self.auxiliary_max_tokens,
            )
            response_dict = extract_json_object_from_text(
                response_text,
                required_key_list=["memories"],
            )
        except Exception as error:
            SafeLogger.warning(f"[Mem0Callback] Failed to extract memories: {error}")
            return
        if not isinstance(response_dict["memories"], list):
            return
        query = extract_question_from_session(session)
        for item in response_dict["memories"]:
            if not isinstance(item, dict):
                continue
            memory_list = merge_unique_texts(
                coerce_str_list(item.get("memory")),
                max_count=1,
            )
            if len(memory_list) == 0:
                continue
            category = str(item.get("category", "fact")).strip() or "fact"
            self._upsert_memory(
                memory_list[0],
                category,
                normalize_entities(coerce_str_list(item.get("entities"))),
                query,
                session.evaluation_record.outcome,
            )
        self._prune_memory()

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_memory_state_path(), self.memory_record_list)
        open(self._get_step_state_path(), "w").write(str(self.current_step))
