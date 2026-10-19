import os
import re
import uuid
from enum import StrEnum
from typing import Any, Optional, Sequence

from pydantic import BaseModel

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import SampleStatus, SessionEvaluationOutcome
from src.utils import SafeLogger

from .advanced_memory_utils import (
    build_trajectory_text,
    coerce_str_list,
    extract_json_object_from_text,
    merge_unique_texts,
)
from .lifelong_memory_utils import (
    bm25_rank,
    build_retrieval_text,
    dump_json_state_list,
    extract_question_from_session,
    inject_memory_block,
    load_json_state_list,
    run_single_prompt,
    truncate_text,
    truncate_text_list_total,
)


class EvoSkillStatus(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    ARCHIVED = "archived"


class EvoSkillAction(StrEnum):
    CREATE = "create"
    EDIT = "edit"


class EvoSkillRecord(BaseModel):
    id: str
    task_name: str
    name: str
    description: str
    triggers: list[str]
    instruction: str
    examples: list[str]
    tags: list[str]
    status: EvoSkillStatus
    action: EvoSkillAction
    parent_skill_id: Optional[str]
    source_failure_queries: list[str]
    created_step: int
    updated_step: int
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    validation_total_count: int = 0
    validation_correct_count: int = 0
    baseline_score: float = 0.0
    validation_score: float = 0.0

    def get_search_text(self) -> str:
        return build_retrieval_text(
            "\n".join(
                [
                    self.task_name,
                    self.name,
                    self.description,
                    " ".join(self.triggers),
                    self.instruction,
                    " ".join(self.examples),
                    " ".join(self.tags),
                    " ".join(self.source_failure_queries),
                ]
            )
        )

    def get_render_text(self, max_instruction_length: int = 1200) -> str:
        trigger_text = "; ".join(self.triggers[:6])
        tag_text = ", ".join(self.tags[:8])
        example_text = "; ".join(self.examples[:2])
        return (
            f"Skill: {self.name}\n"
            f"Description: {self.description}\n"
            f"Triggers: {trigger_text if trigger_text else 'None'}\n"
            f"Tags: {tag_text if tag_text else 'None'}\n"
            f"Procedure:\n{truncate_text(self.instruction, max_instruction_length)}\n"
            f"Examples: {example_text if example_text else 'None'}"
        )


class EvoSkillFailureRecord(BaseModel):
    query: str
    trajectory: str
    retrieved_skill_name_list: list[str]
    outcome: SessionEvaluationOutcome
    created_step: int

    def get_render_text(self, max_trajectory_length: int) -> str:
        skill_text = ", ".join(self.retrieved_skill_name_list)
        return (
            f"Question:\n{self.query}\n"
            f"Outcome: {self.outcome.value}\n"
            f"Active skills: {skill_text if skill_text else 'None'}\n"
            f"Trace:\n{truncate_text(self.trajectory, max_trajectory_length)}"
        )


class EvoSkillFeedbackRecord(BaseModel):
    iteration: int
    action: str
    target_skill_name: Optional[str]
    candidate_skill_name: str
    decision: str
    baseline_score: float
    validation_score: float
    reason: str
    created_step: int

    def get_render_text(self) -> str:
        return (
            f"iter-{self.iteration}: {self.decision.upper()} "
            f"{self.action} {self.candidate_skill_name}; "
            f"target={self.target_skill_name or 'None'}; "
            f"baseline={self.baseline_score:.3f}; "
            f"validation={self.validation_score:.3f}; "
            f"reason={self.reason}"
        )


class EvoSkillCallback(Callback):
    """
    EvoSkill-style online skill discovery for LifelongAgentBench.

    The paper evaluates candidate skill programs on a held-out validation set before
    admitting them to a Pareto frontier. A callback cannot rerun held-out samples
    during the sequential benchmark, so this implementation uses an online analogue:
    each proposed skill is staged as a candidate for the next validation window and
    kept only if its window score beats the recent baseline.
    """

    STOP_WORD_SET = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "be",
        "this",
        "that",
        "these",
        "those",
        "please",
        "task",
        "answer",
        "action",
        "current",
        "similar",
        "future",
        "use",
        "using",
        "from",
        "into",
        "table",
        "headers",
        "return",
    }


    def __init__(
        self,
        language_model: LanguageModel,
        top_k: int = 4,
        max_skill_count: int = 64,
        max_context_length: int = 4000,
        max_trajectory_length: int = 3000,
        failure_batch_size: int = 4,
        evolution_interval: int = 25,
        validation_window: int = 25,
        baseline_window: int = 50,
        feedback_history_size: int = 32,
        max_failure_buffer_size: int = 64,
        min_token_overlap: int = 1,
        min_acceptance_rate: float = 0.0,
        min_score_delta: float = 0.0,
        max_injected_chars_total: int = 4000,
        auxiliary_max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        assert top_k > 0
        assert max_skill_count > 0
        assert max_context_length > 0
        assert max_trajectory_length > 0
        assert failure_batch_size > 0
        assert evolution_interval > 0
        assert validation_window > 0
        assert baseline_window > 0
        assert feedback_history_size > 0
        assert max_failure_buffer_size > 0
        assert min_token_overlap >= 0
        assert 0.0 <= min_acceptance_rate <= 1.0
        assert max_injected_chars_total > 0
        assert auxiliary_max_tokens is None or auxiliary_max_tokens > 0

        self.language_model = language_model
        self.top_k = top_k
        self.max_skill_count = max_skill_count
        self.max_context_length = max_context_length
        self.max_injected_chars_total = max_injected_chars_total
        self.max_trajectory_length = max_trajectory_length
        self.failure_batch_size = failure_batch_size
        self.evolution_interval = evolution_interval
        self.validation_window = validation_window
        self.baseline_window = baseline_window
        self.feedback_history_size = feedback_history_size
        self.max_failure_buffer_size = max_failure_buffer_size
        self.min_token_overlap = min_token_overlap
        self.min_acceptance_rate = min_acceptance_rate
        self.min_score_delta = min_score_delta
        self.auxiliary_max_tokens = auxiliary_max_tokens

        self.skill_list: list[EvoSkillRecord] = []
        self.failure_record_list: list[EvoSkillFailureRecord] = []
        self.feedback_record_list: list[EvoSkillFeedbackRecord] = []
        self.recent_outcome_list: list[int] = []
        self.current_retrieved_skill_index_list: list[int] = []
        self.current_step = 0
        self.iteration = 0

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_skill_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "evoskill_record_list.json")

    def _get_failure_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "evoskill_failure_record_list.json")

    def _get_feedback_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "evoskill_feedback_record_list.json")

    def _get_step_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "evoskill_step.txt")

    def restore_state(self) -> None:
        self.skill_list = load_json_state_list(
            self._get_skill_state_path(),
            EvoSkillRecord,
        )
        self.failure_record_list = load_json_state_list(
            self._get_failure_state_path(),
            EvoSkillFailureRecord,
        )
        self.feedback_record_list = load_json_state_list(
            self._get_feedback_state_path(),
            EvoSkillFeedbackRecord,
        )
        if os.path.exists(self._get_step_state_path()):
            with open(self._get_step_state_path(), "r", encoding="utf-8") as file:
                raw_text = file.read().strip()
            if raw_text:
                parts = raw_text.split()
                self.current_step = int(parts[0])
                self.iteration = int(parts[1]) if len(parts) > 1 else 0

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        self.current_retrieved_skill_index_list = []
        if (
            session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            return

        query = extract_question_from_session(session)
        if len(query.strip()) == 0:
            query = session.chat_history.get_item_deep_copy(-1).content

        selected_index_list = self._rank_skill_index_list(
            query=query,
            task_name=self._get_task_name(callback_args),
            top_k=self.top_k,
        )
        self.current_retrieved_skill_index_list = selected_index_list
        if len(selected_index_list) == 0:
            return

        selected_skill_list = [self.skill_list[index] for index in selected_index_list]
        rendered_skill_list = self._render_skill_context(selected_skill_list)
        for skill in selected_skill_list:
            skill.usage_count += 1
            skill.updated_step = self.current_step

        inject_memory_block(
            session.chat_history,
            (
                "Retrieved EvoSkills. Use a skill only when its triggers match "
                "the current task. The original task instructions and required "
                "action format remain authoritative"
            ),
            truncate_text_list_total(rendered_skill_list, self.max_injected_chars_total),
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        self.current_step += 1
        session = callback_args.current_session

        is_correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
        self._record_active_skill_feedback(is_correct)
        self._update_candidate_validation(is_correct)
        self._record_recent_outcome(is_correct)

        if not is_correct:
            self._record_failure(callback_args)

        if self._has_pending_candidate():
            return
        if self.current_step % self.evolution_interval != 0:
            return
        if len(self.failure_record_list) < self.failure_batch_size:
            return

        self._evolve_from_failures(callback_args)

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        os.makedirs(self.get_state_dir(), exist_ok=True)
        dump_json_state_list(self._get_skill_state_path(), self.skill_list)
        dump_json_state_list(self._get_failure_state_path(), self.failure_record_list)
        dump_json_state_list(self._get_feedback_state_path(), self.feedback_record_list)
        with open(self._get_step_state_path(), "w", encoding="utf-8") as file:
            file.write(f"{self.current_step} {self.iteration}")

    def _get_task_name(self, callback_args: CallbackArguments) -> str:
        value = callback_args.current_session.task_name
        return str(value.value) if hasattr(value, "value") else str(value)

    def _rank_skill_index_list(
        self,
        query: str,
        task_name: str,
        top_k: int,
    ) -> list[int]:
        candidate_pair_list = [
            (index, skill)
            for index, skill in enumerate(self.skill_list)
            if skill.task_name == task_name
            and skill.status in {EvoSkillStatus.ACTIVE, EvoSkillStatus.CANDIDATE}
        ]
        if len(candidate_pair_list) == 0:
            return []

        local_rank_list = bm25_rank(
            build_retrieval_text(query),
            [skill.get_search_text() for _, skill in candidate_pair_list],
            min(top_k * 3, len(candidate_pair_list)),
        )
        selected_index_list: list[int] = []
        for local_index in local_rank_list:
            global_index, skill = candidate_pair_list[local_index]
            if self._token_overlap(query, skill.get_search_text()) < self.min_token_overlap:
                continue
            selected_index_list.append(global_index)
            if len(selected_index_list) >= top_k:
                break
        return selected_index_list

    def _token_overlap(self, query: str, document: str) -> int:
        query_token_set = self._tokenize_for_overlap(query)
        document_token_set = self._tokenize_for_overlap(document)
        if len(query_token_set) == 0 or len(document_token_set) == 0:
            return 0
        return len(query_token_set & document_token_set)

    def _tokenize_for_overlap(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_./-]+", text.lower())
            if len(token) >= 3 and token not in self.STOP_WORD_SET
        }

    def _render_skill_context(self, skill_list: Sequence[EvoSkillRecord]) -> list[str]:
        rendered_list: list[str] = []
        current_length = 0
        for skill in skill_list:
            rendered = skill.get_render_text()
            if current_length + len(rendered) > self.max_context_length:
                remaining_length = self.max_context_length - current_length
                if remaining_length <= 200:
                    break
                rendered = truncate_text(rendered, remaining_length)
            rendered_list.append(rendered)
            current_length += len(rendered)
        return rendered_list

    def _record_active_skill_feedback(self, is_correct: bool) -> None:
        for skill_index in self.current_retrieved_skill_index_list:
            if skill_index >= len(self.skill_list):
                continue
            skill = self.skill_list[skill_index]
            if is_correct:
                skill.success_count += 1
            else:
                skill.failure_count += 1
            skill.updated_step = self.current_step

    def _record_recent_outcome(self, is_correct: bool) -> None:
        self.recent_outcome_list.append(1 if is_correct else 0)
        if len(self.recent_outcome_list) > self.baseline_window:
            self.recent_outcome_list = self.recent_outcome_list[-self.baseline_window :]

    def _get_recent_baseline_score(self) -> float:
        if len(self.recent_outcome_list) == 0:
            return 0.0
        return float(sum(self.recent_outcome_list)) / float(len(self.recent_outcome_list))

    def _record_failure(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        query = extract_question_from_session(session)
        trajectory = build_trajectory_text(
            session,
            callback_args.session_context.agent.get_role_dict(),
            self.max_trajectory_length,
        )
        retrieved_skill_name_list = [
            self.skill_list[index].name
            for index in self.current_retrieved_skill_index_list
            if index < len(self.skill_list)
        ]
        self.failure_record_list.append(
            EvoSkillFailureRecord(
                query=query,
                trajectory=trajectory,
                retrieved_skill_name_list=retrieved_skill_name_list,
                outcome=session.evaluation_record.outcome,
                created_step=self.current_step,
            )
        )
        if len(self.failure_record_list) > self.max_failure_buffer_size:
            self.failure_record_list = self.failure_record_list[
                -self.max_failure_buffer_size :
            ]

    def _has_pending_candidate(self) -> bool:
        return any(skill.status == EvoSkillStatus.CANDIDATE for skill in self.skill_list)

    def _get_pending_candidate(self) -> Optional[EvoSkillRecord]:
        for skill in self.skill_list:
            if skill.status == EvoSkillStatus.CANDIDATE:
                return skill
        return None

    def _update_candidate_validation(self, is_correct: bool) -> None:
        candidate = self._get_pending_candidate()
        if candidate is None:
            return
        candidate.validation_total_count += 1
        if is_correct:
            candidate.validation_correct_count += 1
        candidate.validation_score = (
            float(candidate.validation_correct_count)
            / float(candidate.validation_total_count)
        )
        candidate.updated_step = self.current_step
        if candidate.validation_total_count >= self.validation_window:
            self._finalize_candidate(candidate)

    def _finalize_candidate(self, candidate: EvoSkillRecord) -> None:
        threshold = max(
            self.min_acceptance_rate,
            candidate.baseline_score + self.min_score_delta,
        )
        if (
            self.min_score_delta > 0.0
            or self.min_acceptance_rate > candidate.baseline_score
        ):
            accepted = candidate.validation_score >= threshold
        else:
            accepted = candidate.validation_score > threshold
        if accepted:
            candidate.status = EvoSkillStatus.ACTIVE
            if candidate.parent_skill_id is not None:
                for skill in self.skill_list:
                    if skill.id == candidate.parent_skill_id:
                        skill.status = EvoSkillStatus.ARCHIVED
                        skill.updated_step = self.current_step
                        break
            decision = "accepted"
            reason = "candidate validation score met or exceeded the admission threshold"
        else:
            self._reject_candidate(
                candidate,
                "candidate validation score did not beat the recent baseline",
            )
            return
        self.feedback_record_list.append(
            EvoSkillFeedbackRecord(
                iteration=self.iteration,
                action=candidate.action.value,
                target_skill_name=self._get_parent_skill_name(candidate),
                candidate_skill_name=candidate.name,
                decision=decision,
                baseline_score=candidate.baseline_score,
                validation_score=candidate.validation_score,
                reason=reason,
                created_step=self.current_step,
            )
        )
        self.feedback_record_list = self.feedback_record_list[-self.feedback_history_size :]
        self._prune_skills()

    def _reject_candidate(self, candidate: EvoSkillRecord, reason: str) -> None:
        self.skill_list = [
            skill for skill in self.skill_list if skill.id != candidate.id
        ]
        self.feedback_record_list.append(
            EvoSkillFeedbackRecord(
                iteration=self.iteration,
                action=candidate.action.value,
                target_skill_name=self._get_parent_skill_name(candidate),
                candidate_skill_name=candidate.name,
                decision="rejected",
                baseline_score=candidate.baseline_score,
                validation_score=candidate.validation_score,
                reason=reason,
                created_step=self.current_step,
            )
        )
        self.feedback_record_list = self.feedback_record_list[-self.feedback_history_size :]

    def _get_parent_skill_name(self, candidate: EvoSkillRecord) -> Optional[str]:
        if candidate.parent_skill_id is None:
            return None
        for skill in self.skill_list:
            if skill.id == candidate.parent_skill_id:
                return skill.name
        return None

    def _evolve_from_failures(self, callback_args: CallbackArguments) -> None:
        self.iteration += 1
        failure_batch = self.failure_record_list[-self.failure_batch_size :]
        try:
            proposal_text = run_single_prompt(
                self.language_model,
                self._build_proposer_prompt(callback_args, failure_batch),
                max_tokens=self.auxiliary_max_tokens,
            )
            proposal_dict = extract_json_object_from_text(
                proposal_text,
                required_key_list=["action", "proposed_skill", "justification"],
            )
            builder_text = run_single_prompt(
                self.language_model,
                self._build_builder_prompt(callback_args, proposal_dict, failure_batch),
                max_tokens=self.auxiliary_max_tokens,
            )
            builder_dict = extract_json_object_from_text(
                builder_text,
                required_key_list=[
                    "name",
                    "description",
                    "triggers",
                    "instruction",
                    "tags",
                ],
            )
        except Exception as error:
            SafeLogger.warning(f"[EvoSkillCallback] Failed to evolve skill: {error}")
            return

        candidate = self._build_candidate_skill(
            callback_args=callback_args,
            proposal_dict=proposal_dict,
            builder_dict=builder_dict,
            failure_batch=failure_batch,
        )
        if candidate is None:
            return
        self.skill_list.append(candidate)
        self._prune_skills()
        self.failure_record_list = self.failure_record_list[:-self.failure_batch_size]
        SafeLogger.info(
            "[EvoSkillCallback] Staged candidate skill "
            f"{candidate.name!r} for {self.validation_window} validation samples."
        )

    def _build_candidate_skill(
        self,
        callback_args: CallbackArguments,
        proposal_dict: dict[str, Any],
        builder_dict: dict[str, Any],
        failure_batch: Sequence[EvoSkillFailureRecord],
    ) -> Optional[EvoSkillRecord]:
        raw_action = str(proposal_dict.get("action", "create")).strip().lower()
        action = EvoSkillAction.EDIT if raw_action == "edit" else EvoSkillAction.CREATE
        target_skill_name = str(proposal_dict.get("target_skill", "")).strip()
        parent_skill_id = self._resolve_target_skill_id(target_skill_name)

        name = str(builder_dict.get("name", "")).strip()
        description = str(builder_dict.get("description", "")).strip()
        instruction = str(builder_dict.get("instruction", "")).strip()
        if len(name) == 0 or len(description) == 0 or len(instruction) == 0:
            return None

        existing_id = self._resolve_target_skill_id(name)
        if existing_id is not None and parent_skill_id is None:
            parent_skill_id = existing_id
            action = EvoSkillAction.EDIT

        candidate = EvoSkillRecord(
            id=str(uuid.uuid4()),
            task_name=self._get_task_name(callback_args),
            name=name,
            description=description,
            triggers=merge_unique_texts(coerce_str_list(builder_dict.get("triggers")), 8),
            instruction=instruction,
            examples=merge_unique_texts(coerce_str_list(builder_dict.get("examples")), 3),
            tags=merge_unique_texts(coerce_str_list(builder_dict.get("tags")), 10),
            status=EvoSkillStatus.CANDIDATE,
            action=action,
            parent_skill_id=parent_skill_id,
            source_failure_queries=[failure.query for failure in failure_batch],
            created_step=self.current_step,
            updated_step=self.current_step,
            baseline_score=self._get_recent_baseline_score(),
        )
        return candidate

    def _resolve_target_skill_id(self, target_skill_name: str) -> Optional[str]:
        normalized_target = self._normalize_skill_name(target_skill_name)
        if len(normalized_target) == 0:
            return None
        for skill in self.skill_list:
            if skill.status == EvoSkillStatus.ARCHIVED:
                continue
            if self._normalize_skill_name(skill.name) == normalized_target:
                return skill.id
        return None

    def _normalize_skill_name(self, name: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", name.lower()))

    def _build_proposer_prompt(
        self,
        callback_args: CallbackArguments,
        failure_batch: Sequence[EvoSkillFailureRecord],
    ) -> str:
        active_skill_text = self._render_existing_skills_for_prompt(callback_args)
        feedback_text = self._render_feedback_history()
        failure_text = "\n\n".join(
            f"Failure {index + 1}\n"
            f"{failure.get_render_text(self.max_trajectory_length)}"
            for index, failure in enumerate(failure_batch)
        )
        return f"""You are the Proposer agent in an EvoSkill loop.
Analyze failed agent executions and propose one reusable skill improvement.

Existing active skills:
{active_skill_text if active_skill_text else "None"}

Feedback history:
{feedback_text if feedback_text else "None"}

Failure batch:
{failure_text}

Choose the simplest broad skill that would prevent similar failures.
Prefer editing an existing skill when it should already cover the gap.
Do not propose a table-name-specific, row-value-specific, or one-example-only skill.

Return JSON only:
{{
  "action": "create" or "edit",
  "target_skill": "existing skill name when action is edit, otherwise null",
  "proposed_skill": "high-level capability, inputs, outputs, and workflow",
  "justification": "why this addresses the observed failures and avoids past rejected ideas",
  "related_iterations": ["iter-1"]
}}
"""

    def _build_builder_prompt(
        self,
        callback_args: CallbackArguments,
        proposal_dict: dict[str, Any],
        failure_batch: Sequence[EvoSkillFailureRecord],
    ) -> str:
        task_name = self._get_task_name(callback_args)
        failure_summary = truncate_text("\n\n".join(
            f"- {failure.query}\n{truncate_text(failure.trajectory, 700)}"
            for failure in failure_batch
        ), self.max_trajectory_length)
        proposal_text = truncate_text(str(proposal_dict), self.max_trajectory_length)
        return f"""You are the Skill-Builder agent in an EvoSkill loop.
Materialize the proposal into a concise skill that can be injected into a coding agent prompt.

Task environment: {task_name}

Proposal:
{proposal_text}

Failure evidence:
{failure_summary}

Build a reusable skill with explicit triggers and a procedural checklist.
Keep it general across tables, schemas, files, and concrete values.
The skill must preserve the benchmark's required action format.

Return JSON only:
{{
  "name": "short_snake_case_or_kebab_case_name",
  "description": "one sentence capability description",
  "triggers": ["when to use this skill"],
  "instruction": "concise procedural checklist the executor should follow",
  "tags": ["domain", "operation", "failure-mode"],
  "examples": ["optional compact example"]
}}
"""

    def _render_existing_skills_for_prompt(self, callback_args: CallbackArguments) -> str:
        task_name = self._get_task_name(callback_args)
        active_skill_list = [
            skill
            for skill in self.skill_list
            if skill.task_name == task_name and skill.status == EvoSkillStatus.ACTIVE
        ]
        return "\n\n".join(
            f"Name: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"Triggers: {'; '.join(skill.triggers[:5])}\n"
            f"Recent score: {self._calculate_skill_score(skill):.3f}"
            for skill in active_skill_list[-self.max_skill_count :]
        )

    def _render_feedback_history(self) -> str:
        return "\n".join(
            feedback.get_render_text()
            for feedback in self.feedback_record_list[-self.feedback_history_size :]
        )

    def _calculate_skill_score(self, skill: EvoSkillRecord) -> float:
        total = skill.success_count + skill.failure_count
        if total == 0:
            return skill.validation_score
        return float(skill.success_count) / float(total)

    def _prune_skills(self) -> None:
        # The bank budget includes pending candidates. Archived skill bodies
        # cannot remain as an unbounded hidden persistent memory repository.
        candidates = [skill for skill in self.skill_list if skill.status == EvoSkillStatus.CANDIDATE]
        if len(candidates) > self.max_skill_count:
            raise ValueError("EvoSkill candidates exceed the persistent memory budget.")
        active = [skill for skill in self.skill_list if skill.status == EvoSkillStatus.ACTIVE]
        parent_ids = {skill.parent_skill_id for skill in candidates if skill.parent_skill_id is not None}
        active.sort(key=lambda skill: (skill.id in parent_ids, self._calculate_skill_score(skill), skill.usage_count, skill.updated_step), reverse=True)
        retained_ids = {skill.id for skill in candidates + active[:self.max_skill_count-len(candidates)]}
        self.skill_list = [skill for skill in self.skill_list if skill.id in retained_ids]
