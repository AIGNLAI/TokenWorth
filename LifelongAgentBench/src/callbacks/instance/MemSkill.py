import os
import re
from enum import StrEnum
from typing import Optional, Sequence

from pydantic import BaseModel

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import SampleStatus, SessionEvaluationOutcome
from src.utils import SafeLogger
from .advanced_memory_utils import (
    extract_json_object_from_text,
)
from .lifelong_memory_utils import (
    bm25_rank,
    build_retrieval_text,
    dump_json_state_list,
    extract_question_from_session,
    inject_memory_block,
    load_json_state_list,
    normalize_memory_text,
    run_single_prompt,
    truncate_text,
)


class MemSkillUpdateType(StrEnum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


class MemSkill(BaseModel):
    name: str
    description: str
    instruction_template: str
    update_type: MemSkillUpdateType
    created_step: int
    use_count: int
    success_count: int
    failure_count: int

    def get_render_text(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"{self.instruction_template}"
        )

    def get_search_text(self) -> str:
        return build_retrieval_text(f"{self.description}\n{self.instruction_template}")

    def get_success_rate(self) -> float:
        return (float(self.success_count) + 1.0) / (float(self.use_count) + 2.0)


class MemSkillMemoryRecord(BaseModel):
    memory: str
    source_query: str
    source_outcome: SessionEvaluationOutcome
    created_step: int
    last_access_step: int
    access_count: int
    strength: float

    def get_search_text(self) -> str:
        return build_retrieval_text(f"{self.source_query}\n{self.memory}")


class MemSkillHardCase(BaseModel):
    query: str
    trajectory: str
    retrieved_memory_list: list[str]
    selected_skill_name_list: list[str]
    outcome: SessionEvaluationOutcome
    created_step: int
    fail_count: int


class MemSkillAction(BaseModel):
    action: MemSkillUpdateType
    memory_item: Optional[str] = None
    memory_index: Optional[int] = None
    updated_memory: Optional[str] = None


_ACTION_PATTERN = re.compile(
    r"^\s*ACTION:\s*(INSERT|UPDATE|DELETE|NOOP)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FIELD_PATTERN_TEMPLATE = r"^\s*{field_name}:\s*([\s\S]*?)(?=^\s*[A-Z][A-Z ]+:\s*|\Z)"


def _normalize_skill_name(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


def _extract_block_field(block: str, field_name: str) -> str:
    pattern = re.compile(
        _FIELD_PATTERN_TEMPLATE.format(field_name=re.escape(field_name)),
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(block)
    if match is None:
        return ""
    return match.group(1).strip()


def _parse_memory_index(raw_text: str) -> Optional[int]:
    match = re.search(r"-?\d+", raw_text)
    if match is None:
        return None
    value = int(match.group(0))
    return value if value >= 0 else None


class MemSkillCallback(Callback):
    def __init__(
        self,
        language_model: LanguageModel,
        top_k: int,
        top_k_skills: int = 3,
        max_memory_count: int = 512,
        max_skill_count: int = 24,
        max_trajectory_length: int = 3000,
        skill_evolution_interval: int = 0,
        hard_case_buffer_size: int = 32,
        max_evolution_changes: int = 2,
        correct_memory_only: bool = True,
        max_injected_memory_chars_per_item: int = 400,
        max_injected_memory_chars_total: int = 1600,
        max_injected_chars_total: int | None = None,
        retrieval_strength_weight: float = 0.5,
        retrieval_access_weight: float = 0.0,
        max_retrieval_strength: float = 3.0,
        max_retrieval_access_count: int = 5,
        use_evaluator_outcome_only: bool = False,
        auxiliary_max_tokens: int | None = None,
    ):
        super().__init__()
        assert top_k > 0
        assert top_k_skills > 0
        assert max_memory_count > 0
        assert max_skill_count >= 4
        assert skill_evolution_interval >= 0
        assert hard_case_buffer_size > 0
        assert max_evolution_changes > 0
        assert max_injected_memory_chars_per_item > 0
        if max_injected_chars_total is not None:
            max_injected_memory_chars_total = max_injected_chars_total
        assert max_injected_memory_chars_total > 0
        assert retrieval_strength_weight >= 0
        assert retrieval_access_weight >= 0
        assert max_retrieval_strength >= 0
        assert max_retrieval_access_count >= 0
        assert auxiliary_max_tokens is None or auxiliary_max_tokens > 0
        self.language_model = language_model
        self.top_k = top_k
        self.top_k_skills = top_k_skills
        self.max_memory_count = max_memory_count
        self.max_skill_count = max_skill_count
        self.max_trajectory_length = max_trajectory_length
        self.skill_evolution_interval = skill_evolution_interval
        self.hard_case_buffer_size = hard_case_buffer_size
        self.max_evolution_changes = max_evolution_changes
        self.correct_memory_only = correct_memory_only
        self.max_injected_memory_chars_per_item = max_injected_memory_chars_per_item
        self.max_injected_memory_chars_total = max_injected_memory_chars_total
        self.retrieval_strength_weight = retrieval_strength_weight
        self.retrieval_access_weight = retrieval_access_weight
        self.max_retrieval_strength = max_retrieval_strength
        self.max_retrieval_access_count = max_retrieval_access_count
        self.use_evaluator_outcome_only = use_evaluator_outcome_only
        self.auxiliary_max_tokens = auxiliary_max_tokens
        self.memory_record_list: list[MemSkillMemoryRecord] = []
        self.skill_list: list[MemSkill] = self._build_default_skill_list()
        self.hard_case_list: list[MemSkillHardCase] = []
        self.current_step = 0
        self.current_retrieved_record_index_list: list[int] = []
        self.current_selected_skill_index_list: list[int] = []

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_memory_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memskill_memory_record_list.json")

    def _get_skill_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memskill_skill_list.json")

    def _get_hard_case_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memskill_hard_case_list.json")

    def _get_step_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memskill_step.txt")

    def restore_state(self) -> None:
        self.memory_record_list = [
            MemSkillMemoryRecord.model_validate(item)
            for item in load_json_state_list(
                self._get_memory_state_path(), MemSkillMemoryRecord
            )
        ]
        restored_skill_list = [
            MemSkill.model_validate(item)
            for item in load_json_state_list(self._get_skill_state_path(), MemSkill)
        ]
        if len(restored_skill_list) > 0:
            self.skill_list = restored_skill_list
        self.hard_case_list = [
            MemSkillHardCase.model_validate(item)
            for item in load_json_state_list(
                self._get_hard_case_state_path(), MemSkillHardCase
            )
        ]
        if os.path.exists(self._get_step_state_path()):
            self.current_step = int(
                open(self._get_step_state_path(), "r").read().strip()
            )

    def _build_default_skill_list(self) -> list[MemSkill]:
        return [
            MemSkill(
                name="insert_new_memory",
                description=(
                    "Capture new durable facts, constraints, procedures, or lessons "
                    "from the current text chunk that are missing in memory."
                ),
                instruction_template=(
                    "Skill: Insert New Memory\n"
                    "Purpose: Capture new, durable information from the text chunk.\n"
                    "When to use:\n"
                    "- The text chunk introduces a new fact, event, plan, constraint, "
                    "or reusable procedure worth storing.\n"
                    "- The information is stable and likely useful later.\n"
                    "How to apply:\n"
                    "- Compare against retrieved memories to avoid duplicates.\n"
                    "- Split distinct facts into separate memory items.\n"
                    "- Keep each item concise, specific, and self-contained.\n"
                    "Constraints:\n"
                    "- Skip trivial, fleeting, speculative, or sample-only details.\n"
                    "- Do not update or delete existing memories.\n"
                    "Action type: INSERT only."
                ),
                update_type=MemSkillUpdateType.INSERT,
                created_step=0,
                use_count=0,
                success_count=0,
                failure_count=0,
            ),
            MemSkill(
                name="update_existing_memory",
                description=(
                    "Revise a retrieved memory with new, corrected, or more specific "
                    "information from the current text chunk."
                ),
                instruction_template=(
                    "Skill: Update Existing Memory\n"
                    "Purpose: Revise a retrieved memory with new or corrected details.\n"
                    "When to use:\n"
                    "- The text chunk clarifies, corrects, or extends a retrieved memory.\n"
                    "- A retrieved memory is still mostly valid but incomplete.\n"
                    "How to apply:\n"
                    "- Select the best matching retrieved memory item.\n"
                    "- Merge new details into one concise updated memory.\n"
                    "- Preserve accurate details that still hold.\n"
                    "Constraints:\n"
                    "- Do not create new memories.\n"
                    "- Do not delete memory items.\n"
                    "Action type: UPDATE only."
                ),
                update_type=MemSkillUpdateType.UPDATE,
                created_step=0,
                use_count=0,
                success_count=0,
                failure_count=0,
            ),
            MemSkill(
                name="delete_invalid_memory",
                description=(
                    "Remove a retrieved memory that is explicitly wrong, outdated, "
                    "contradicted, or superseded by the current text chunk."
                ),
                instruction_template=(
                    "Skill: Delete Invalid Memory\n"
                    "Purpose: Remove retrieved memory that is wrong or obsolete.\n"
                    "When to use:\n"
                    "- The text chunk clearly contradicts a retrieved memory.\n"
                    "- A plan, fact, or rule is explicitly canceled or replaced.\n"
                    "How to apply:\n"
                    "- Delete only the retrieved memory item directly contradicted.\n"
                    "Constraints:\n"
                    "- If uncertain, prefer no action over deletion.\n"
                    "Action type: DELETE only."
                ),
                update_type=MemSkillUpdateType.DELETE,
                created_step=0,
                use_count=0,
                success_count=0,
                failure_count=0,
            ),
            MemSkill(
                name="no_operation",
                description=(
                    "Confirm that no memory update is needed when the text chunk has "
                    "no new, corrective, or durable information."
                ),
                instruction_template=(
                    "Skill: No Operation\n"
                    "Purpose: Confirm no memory changes are needed for the text chunk.\n"
                    "When to use:\n"
                    "- The text chunk contains no new, corrective, or actionable "
                    "information worth storing.\n"
                    "Constraints:\n"
                    "- Emit NOOP only if none of the selected skills produce actions.\n"
                    "Action type: NOOP only."
                ),
                update_type=MemSkillUpdateType.NOOP,
                created_step=0,
                use_count=0,
                success_count=0,
                failure_count=0,
            ),
        ]

    def _build_trajectory_text(self, callback_args: CallbackArguments) -> str:
        session = callback_args.current_session
        role_dict = callback_args.session_context.agent.get_role_dict()
        if session.chat_history.get_value_length() <= 3:
            return ""
        return truncate_text(
            session.chat_history.get_value_str(
                role_dict,
                start_index=3,
                end_index=None,
            ),
            self.max_trajectory_length,
        )

    def _select_memory_index_list(self, query: str) -> list[int]:
        if len(self.memory_record_list) == 0:
            return []
        candidate_index_list = [
            index
            for index, record in enumerate(self.memory_record_list)
            if (
                not self.correct_memory_only
                or record.source_outcome == SessionEvaluationOutcome.CORRECT
            )
        ]
        if len(candidate_index_list) == 0:
            return []
        lexical_rank_index_list = bm25_rank(
            build_retrieval_text(query),
            [
                self.memory_record_list[index].get_search_text()
                for index in candidate_index_list
            ],
            len(candidate_index_list),
        )
        candidate_count = float(len(candidate_index_list))
        lexical_bonus_dict = {
            candidate_index_list[relative_index]: 4.0
            * float(len(candidate_index_list) - rank)
            / candidate_count
            for rank, relative_index in enumerate(lexical_rank_index_list)
        }
        score_and_index_list: list[tuple[float, int]] = []
        for record_index in candidate_index_list:
            record = self.memory_record_list[record_index]
            recency_gap = max(0, self.current_step - record.last_access_step)
            recency_bonus = 1.0 / float(recency_gap + 1)
            outcome_bonus = (
                1.0
                if record.source_outcome == SessionEvaluationOutcome.CORRECT
                else -2.0
            )
            strength_bonus = min(record.strength, self.max_retrieval_strength)
            access_bonus = min(record.access_count, self.max_retrieval_access_count)
            score = (
                lexical_bonus_dict.get(record_index, 0.0)
                + self.retrieval_strength_weight * strength_bonus
                + self.retrieval_access_weight * float(access_bonus)
                + recency_bonus
                + outcome_bonus
            )
            score_and_index_list.append((score, record_index))
        score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [index for _, index in score_and_index_list[: self.top_k]]

    def _build_injected_memory_list(self, record_index_list: Sequence[int]) -> list[str]:
        injected_memory_list: list[str] = []
        remaining_chars = self.max_injected_memory_chars_total
        for index in record_index_list:
            if index >= len(self.memory_record_list) or remaining_chars <= 0:
                continue
            memory = truncate_text(
                self.memory_record_list[index].memory,
                min(self.max_injected_memory_chars_per_item, remaining_chars),
            )
            memory = memory.strip()
            if not memory:
                continue
            injected_memory_list.append(memory)
            remaining_chars -= len(memory)
        if len(injected_memory_list) > 0:
            injected_memory_list.append(
                "Use retrieved memories only when they are relevant to the current "
                "question; ignore unrelated names, literal values, and examples."
            )
        return injected_memory_list

    def _is_memory_text_allowed(self, memory_text: str) -> bool:
        normalized_text = normalize_memory_text(memory_text)
        if len(normalized_text) == 0:
            return False
        if "executed successfully" in normalized_text:
            return False
        if "final answer" in normalized_text:
            return False
        return True

    def _select_skill_index_list(
        self,
        text_span: str,
        retrieved_memory_list: Sequence[str],
    ) -> list[int]:
        context_text = f"{text_span}\n" + "\n".join(retrieved_memory_list)
        lexical_rank_index_list = bm25_rank(
            build_retrieval_text(context_text),
            [skill.get_search_text() for skill in self.skill_list],
            len(self.skill_list),
        )
        lexical_bonus_dict = {
            skill_index: 4.0
            * float(len(self.skill_list) - rank)
            / float(len(self.skill_list))
            for rank, skill_index in enumerate(lexical_rank_index_list)
        }
        score_and_index_list: list[tuple[float, int]] = []
        for skill_index, skill in enumerate(self.skill_list):
            exploration_bonus = 0.4 if skill.use_count == 0 else 0.0
            score = (
                lexical_bonus_dict.get(skill_index, 0.0)
                + 1.5 * skill.get_success_rate()
                + exploration_bonus
            )
            score_and_index_list.append((score, skill_index))
        score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected_index_list: list[int] = []
        preferred_skill_name_list = ["insert_new_memory"]
        if len(retrieved_memory_list) > 0:
            preferred_skill_name_list.append("update_existing_memory")
        preferred_skill_name_list.append("no_operation")
        for preferred_skill_name in preferred_skill_name_list:
            for skill_index, skill in enumerate(self.skill_list):
                if skill.name != preferred_skill_name:
                    continue
                if skill_index not in selected_index_list:
                    selected_index_list.append(skill_index)
                break
            if len(selected_index_list) >= self.top_k_skills:
                return selected_index_list
        for _, skill_index in score_and_index_list:
            if skill_index in selected_index_list:
                continue
            selected_index_list.append(skill_index)
            if len(selected_index_list) >= self.top_k_skills:
                break
        return selected_index_list

    def _build_executor_prompt(
        self,
        text_span: str,
        retrieved_memory_list: Sequence[str],
        selected_skill_list: Sequence[MemSkill],
    ) -> str:
        memory_text = "\n".join(
            f"{index}. {memory}"
            for index, memory in enumerate(retrieved_memory_list)
        )
        skill_text = "\n\n".join(skill.get_render_text() for skill in selected_skill_list)
        return f"""You are a memory management executor.
Apply selected skills to the input text chunk and retrieved memories.
Then output memory actions.

Input Text Chunk:
{text_span}

Retrieved Memories (0-based index):
{memory_text if memory_text else "None"}

Selected Skills:
{skill_text}

Guidelines:
- Apply any skill as needed; a skill may be used multiple times.
- Read the input text chunk carefully line by line and apply any skill as needed.
- Store reusable patterns, constraints, procedures, or pitfalls rather than sample-specific outputs.
- Keep each memory atomic. Do not merge unrelated tasks into one memory.
- Only use action types supported by the selected skills.
- MEMORY INDEX is 0-based and must reference the retrieved memories list.
- Output only action blocks in the format below.
- Do not include explanations or reasoning lines.

Output format. Use one block per action and separate blocks with a blank line:
INSERT block:
ACTION: INSERT
MEMORY ITEM: [one concise reusable memory item, preferably under 80 words]

UPDATE block:
ACTION: UPDATE
MEMORY INDEX: [0-based index]
UPDATED MEMORY: [concise but complete merged summary with essential updates]

DELETE block:
ACTION: DELETE
MEMORY INDEX: [0-based index]

NOOP block:
ACTION: NOOP
"""

    def _parse_executor_actions(self, raw_text: str) -> list[MemSkillAction]:
        action_match_list = list(_ACTION_PATTERN.finditer(raw_text))
        action_list: list[MemSkillAction] = []
        for match_index, match in enumerate(action_match_list):
            block_start = match.start()
            block_end = (
                action_match_list[match_index + 1].start()
                if match_index + 1 < len(action_match_list)
                else len(raw_text)
            )
            block = raw_text[block_start:block_end]
            action = MemSkillUpdateType(match.group(1).lower())
            if action == MemSkillUpdateType.INSERT:
                memory_item = _extract_block_field(block, "MEMORY ITEM")
                if memory_item:
                    action_list.append(
                        MemSkillAction(action=action, memory_item=memory_item)
                    )
            elif action == MemSkillUpdateType.UPDATE:
                memory_index = _parse_memory_index(
                    _extract_block_field(block, "MEMORY INDEX")
                )
                updated_memory = _extract_block_field(block, "UPDATED MEMORY")
                if memory_index is not None and updated_memory:
                    action_list.append(
                        MemSkillAction(
                            action=action,
                            memory_index=memory_index,
                            updated_memory=updated_memory,
                        )
                    )
            elif action == MemSkillUpdateType.DELETE:
                memory_index = _parse_memory_index(
                    _extract_block_field(block, "MEMORY INDEX")
                )
                if memory_index is not None:
                    action_list.append(
                        MemSkillAction(action=action, memory_index=memory_index)
                    )
            else:
                action_list.append(MemSkillAction(action=action))
        return action_list

    def _upsert_memory(
        self,
        memory_text: str,
        query: str,
        outcome: SessionEvaluationOutcome,
    ) -> None:
        if outcome != SessionEvaluationOutcome.CORRECT and self.correct_memory_only:
            return
        memory_text = truncate_text(
            memory_text.strip(),
            self.max_injected_memory_chars_per_item,
        )
        if not self._is_memory_text_allowed(memory_text):
            return
        normalized_memory = normalize_memory_text(memory_text)
        if len(normalized_memory) == 0:
            return
        for record in self.memory_record_list:
            if normalize_memory_text(record.memory) != normalized_memory:
                continue
            record.access_count += 1
            record.last_access_step = self.current_step
            record.strength = min(
                record.strength + (0.4 if outcome == SessionEvaluationOutcome.CORRECT else 0.2),
                self.max_retrieval_strength,
            )
            if len(query) > len(record.source_query):
                record.source_query = query
            if outcome == SessionEvaluationOutcome.CORRECT:
                record.source_outcome = outcome
            return
        self.memory_record_list.append(
            MemSkillMemoryRecord(
                memory=memory_text.strip(),
                source_query=query,
                source_outcome=outcome,
                created_step=self.current_step,
                last_access_step=self.current_step,
                access_count=1,
                strength=1.0 if outcome == SessionEvaluationOutcome.CORRECT else 0.6,
            )
        )

    def _apply_executor_actions(
        self,
        action_list: Sequence[MemSkillAction],
        retrieved_record_index_list: Sequence[int],
        query: str,
        outcome: SessionEvaluationOutcome,
    ) -> None:
        if outcome != SessionEvaluationOutcome.CORRECT and self.correct_memory_only:
            return
        deleted_record_index_set: set[int] = set()
        for action in action_list:
            if action.action == MemSkillUpdateType.INSERT and action.memory_item:
                self._upsert_memory(action.memory_item, query, outcome)
                continue
            if (
                action.action == MemSkillUpdateType.UPDATE
                and action.memory_index is not None
                and action.updated_memory
                and action.memory_index < len(retrieved_record_index_list)
            ):
                record_index = retrieved_record_index_list[action.memory_index]
                if record_index in deleted_record_index_set:
                    continue
                record = self.memory_record_list[record_index]
                updated_memory = truncate_text(
                    action.updated_memory.strip(),
                    self.max_injected_memory_chars_per_item,
                )
                if not self._is_memory_text_allowed(updated_memory):
                    continue
                record.memory = updated_memory
                if len(query) > len(record.source_query):
                    record.source_query = query
                record.source_outcome = outcome
                record.last_access_step = self.current_step
                record.access_count += 1
                record.strength = min(
                    record.strength + (0.5 if outcome == SessionEvaluationOutcome.CORRECT else 0.1),
                    self.max_retrieval_strength,
                )
                continue
            if (
                action.action == MemSkillUpdateType.DELETE
                and action.memory_index is not None
                and action.memory_index < len(retrieved_record_index_list)
            ):
                deleted_record_index_set.add(retrieved_record_index_list[action.memory_index])
        if len(deleted_record_index_set) > 0:
            self.memory_record_list = [
                record
                for index, record in enumerate(self.memory_record_list)
                if index not in deleted_record_index_set
            ]

    def _prune_memory(self) -> None:
        if len(self.memory_record_list) <= self.max_memory_count:
            return
        self.memory_record_list.sort(
            key=lambda record: (
                record.strength,
                record.access_count,
                record.last_access_step,
            ),
            reverse=True,
        )
        self.memory_record_list = self.memory_record_list[: self.max_memory_count]

    def _record_skill_feedback(self, outcome: SessionEvaluationOutcome) -> None:
        success = outcome == SessionEvaluationOutcome.CORRECT
        for skill_index in self.current_selected_skill_index_list:
            skill = self.skill_list[skill_index]
            skill.use_count += 1
            if success:
                skill.success_count += 1
            else:
                skill.failure_count += 1

    def _record_hard_case(
        self,
        query: str,
        trajectory: str,
        retrieved_memory_list: Sequence[str],
        selected_skill_list: Sequence[MemSkill],
        outcome: SessionEvaluationOutcome,
    ) -> None:
        if outcome == SessionEvaluationOutcome.CORRECT:
            return
        hard_case = MemSkillHardCase(
            query=query,
            trajectory=trajectory,
            retrieved_memory_list=list(retrieved_memory_list),
            selected_skill_name_list=[skill.name for skill in selected_skill_list],
            outcome=outcome,
            created_step=self.current_step,
            fail_count=1,
        )
        self.hard_case_list.append(hard_case)
        if len(self.hard_case_list) > self.hard_case_buffer_size:
            self.hard_case_list = self.hard_case_list[-self.hard_case_buffer_size :]

    def _build_designer_prompt(self) -> str:
        skill_bank_text = "\n\n".join(skill.get_render_text() for skill in self.skill_list)
        ranked_hard_case_list = sorted(
            self.hard_case_list,
            key=lambda hard_case: (hard_case.fail_count, hard_case.created_step),
            reverse=True,
        )[: min(6, len(self.hard_case_list))]
        hard_case_text_list: list[str] = []
        for case_index, hard_case in enumerate(ranked_hard_case_list, start=1):
            memory_text = "\n".join(f"- {item}" for item in hard_case.retrieved_memory_list)
            skill_text = ", ".join(hard_case.selected_skill_name_list)
            hard_case_text_list.append(
                f"Case {case_index}\n"
                f"Question: {hard_case.query}\n"
                f"Outcome: {hard_case.outcome.value}\n"
                f"Selected skills: {skill_text}\n"
                f"Retrieved memories:\n{memory_text if memory_text else '- None'}\n"
                f"Trajectory:\n{truncate_text(hard_case.trajectory, 1200)}"
            )
        hard_case_text = "\n\n".join(hard_case_text_list)
        return f"""You are a designer for a MemSkill memory system.
Improve the memory skill bank by inspecting hard cases.
Focus on cases where selected skills produced incomplete or unhelpful memories.

Current Operation Bank:
{skill_bank_text}

Failure Cases:
{hard_case_text}

Task:
Propose up to {self.max_evolution_changes} improvements.
- Add a new operation only if the failures show a capability gap.
- Refine an existing operation if its instruction is too vague or misses important details.
- Only propose operations with update type "insert" or "update".
- Do not modify delete or noop operations.
- Instruction templates must clearly state purpose, triggers, steps, constraints, and action type.
- Avoid marketing labels such as enhanced or advanced.

Return JSON only:
{{
  "action": "apply changes",
  "summary": "overall rationale",
  "changes": [
    {{
      "action": "add new",
      "new operation": {{
        "name": "snake_case_name",
        "description": "what it does and when to use it",
        "instruction template": "Skill: ...",
        "update type": "insert",
        "reasoning": "why this helps"
      }}
    }},
    {{
      "action": "refine existing",
      "refined operation": {{
        "name": "existing operation name",
        "changes": {{
          "description": "improved description",
          "instruction template": "Skill: ..."
        }},
        "reasoning": "why this helps"
      }}
    }}
  ]
}}

If no change is justified, return:
{{"action": "no change", "reasoning": "why"}}
"""

    def _apply_skill_evolution_change_dict(self, change_dict: dict) -> None:
        action = str(change_dict.get("action", "")).strip().lower()
        if action == "add new":
            if len(self.skill_list) >= self.max_skill_count:
                return
            operation_dict = change_dict.get("new operation")
            if not isinstance(operation_dict, dict):
                return
            update_type_text = str(operation_dict.get("update type", "")).strip().lower()
            if update_type_text not in {
                MemSkillUpdateType.INSERT.value,
                MemSkillUpdateType.UPDATE.value,
            }:
                return
            name = str(operation_dict.get("name", "")).strip()
            description = str(operation_dict.get("description", "")).strip()
            instruction_template = str(
                operation_dict.get("instruction template", "")
            ).strip()
            if not name or not description or not instruction_template:
                return
            existing_name_set = {_normalize_skill_name(skill.name) for skill in self.skill_list}
            if _normalize_skill_name(name) in existing_name_set:
                return
            self.skill_list.append(
                MemSkill(
                    name=name,
                    description=description,
                    instruction_template=instruction_template,
                    update_type=MemSkillUpdateType(update_type_text),
                    created_step=self.current_step,
                    use_count=0,
                    success_count=0,
                    failure_count=0,
                )
            )
            return
        if action != "refine existing":
            return
        operation_dict = change_dict.get("refined operation")
        if not isinstance(operation_dict, dict):
            return
        target_name = _normalize_skill_name(str(operation_dict.get("name", "")))
        changes_dict = operation_dict.get("changes")
        if not isinstance(changes_dict, dict):
            return
        for skill in self.skill_list:
            if _normalize_skill_name(skill.name) != target_name:
                continue
            if skill.update_type not in {
                MemSkillUpdateType.INSERT,
                MemSkillUpdateType.UPDATE,
            }:
                return
            description = str(changes_dict.get("description", "")).strip()
            instruction_template = str(
                changes_dict.get("instruction template", "")
            ).strip()
            if description:
                skill.description = description
            if instruction_template:
                skill.instruction_template = instruction_template
            return

    def _maybe_evolve_skills(self) -> None:
        if self.skill_evolution_interval <= 0:
            return
        if self.current_step == 0 or self.current_step % self.skill_evolution_interval != 0:
            return
        if len(self.hard_case_list) == 0:
            return
        try:
            response_text = run_single_prompt(
                self.language_model,
                self._build_designer_prompt(),
                max_tokens=self.auxiliary_max_tokens,
            )
            response_dict = extract_json_object_from_text(response_text)
        except Exception as error:
            SafeLogger.warning(f"[MemSkillCallback] Failed to evolve skills: {error}")
            return
        if str(response_dict.get("action", "")).strip().lower() != "apply changes":
            return
        change_list = response_dict.get("changes")
        if not isinstance(change_list, Sequence) or isinstance(change_list, str):
            return
        for change_dict in change_list[: self.max_evolution_changes]:
            if isinstance(change_dict, dict):
                self._apply_skill_evolution_change_dict(change_dict)

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            self.current_retrieved_record_index_list = []
            return
        query = session.chat_history.get_item_deep_copy(-1).content
        selected_index_list = self._select_memory_index_list(query)
        for index in selected_index_list:
            record = self.memory_record_list[index]
            record.last_access_step = self.current_step
            record.access_count += 1
        self.current_retrieved_record_index_list = selected_index_list
        inject_memory_block(
            session.chat_history,
            "MemSkill memories",
            self._build_injected_memory_list(selected_index_list),
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
        trajectory = self._build_trajectory_text(callback_args)
        text_span = (
            f"Task:\n{query}\n\n"
            f"Outcome:\n{session.evaluation_record.outcome.value}\n\n"
            f"Trajectory:\n{trajectory}"
        )
        retrieved_index_list = [
            index
            for index in self.current_retrieved_record_index_list
            if index < len(self.memory_record_list)
        ]
        if len(retrieved_index_list) == 0:
            retrieved_index_list = self._select_memory_index_list(query)
        retrieved_memory_list = [
            self.memory_record_list[index].memory for index in retrieved_index_list
        ]
        selected_skill_index_list = self._select_skill_index_list(
            text_span,
            retrieved_memory_list,
        )
        self.current_selected_skill_index_list = selected_skill_index_list
        selected_skill_list = [self.skill_list[index] for index in selected_skill_index_list]
        try:
            response_text = run_single_prompt(
                self.language_model,
                self._build_executor_prompt(
                    truncate_text(text_span, self.max_trajectory_length),
                    retrieved_memory_list,
                    selected_skill_list,
                ),
                max_tokens=self.auxiliary_max_tokens,
            )
            action_list = self._parse_executor_actions(response_text)
        except Exception as error:
            SafeLogger.warning(f"[MemSkillCallback] Failed to execute skills: {error}")
            action_list = []
        self._apply_executor_actions(
            action_list,
            retrieved_index_list,
            query,
            session.evaluation_record.outcome,
        )
        self._record_skill_feedback(session.evaluation_record.outcome)
        self._record_hard_case(
            query,
            trajectory,
            retrieved_memory_list,
            selected_skill_list,
            session.evaluation_record.outcome,
        )
        self._prune_memory()
        self._maybe_evolve_skills()
        self.current_retrieved_record_index_list = []
        self.current_selected_skill_index_list = []

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_memory_state_path(), self.memory_record_list)
        dump_json_state_list(self._get_skill_state_path(), self.skill_list)
        dump_json_state_list(self._get_hard_case_state_path(), self.hard_case_list)
        open(self._get_step_state_path(), "w").write(str(self.current_step))
