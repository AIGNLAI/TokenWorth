import hashlib
import json
import math
import os
import re
import statistics
from typing import Sequence

from pydantic import BaseModel

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import ChatHistory, ChatHistoryItem, Role, SampleStatus, Session
from src.typings import SessionEvaluationOutcome

from .lifelong_memory_utils import (
    dump_json_state_list,
    extract_question_from_session,
    load_json_state_list,
    rank_texts_by_relevance,
    tokenize,
    truncate_text,
)


_ACTION_PATTERN_LIST = [
    re.compile(r"Action:\s*[A-Za-z_]\w*\([^\r\n]*\)", re.I),
    re.compile(r"Final\s+Answer:\s*(?:Var(?:iable)?\s*)?#\d+", re.I),
    re.compile(r"Action:\s*Operation[\s\S]*?(?:```[\s\S]*?```)", re.I),
    re.compile(r"Action:\s*Answer[\s\S]*", re.I),
    re.compile(r"Act:\s*bash[\s\S]*?(?:```[\s\S]*?```)", re.I),
    re.compile(r"Act:\s*finish[\s\S]*", re.I),
]


class TokenWorthMemoryRecord(BaseModel):
    memory_id: int
    query: str
    snippet: str
    search_text: str
    query_embedding: list[float]
    signature: str
    created_step: int
    last_access_step: int
    access_count: int = 0
    help_count: int = 0
    harm_count: int = 0
    utility: float = 1.0

    def reliability(self) -> float:
        return (self.help_count + 1.0) / (
            self.help_count + self.harm_count + 2.0
        )

    def render(self, max_chars: int) -> str:
        return truncate_text(self.snippet, max_chars)


class TokenWorthPairRecord(BaseModel):
    pair_feature: list[float]
    label: float
    memory_id: int
    created_step: int


class TokenWorthBaseFeedbackRecord(BaseModel):
    feature: list[float]
    label: float


class TokenWorthMetrics(BaseModel):
    step: int = 0
    retrieve_count: int = 0
    skip_retrieve_count: int = 0
    gate_skip_count: int = 0
    write_count: int = 0
    replace_count: int = 0
    evict_count: int = 0
    pair_write_count: int = 0
    rfm_update_count: int = 0
    solved_by_existing_memory_count: int = 0
    novelty_write_count: int = 0
    conflict_replace_count: int = 0
    positive_pair_count: int = 0
    negative_pair_count: int = 0


class TokenWorthBaseMemoryCallback(Callback):
    def __init__(
        self,
        embedding_model_path: str,
        embedding_device: str = "cpu",
        embedding_max_length: int = 192,
        embedding_cache_enabled: bool = True,
        top_k: int = 1,
        max_memory_count: int = 32,
        max_snippet_length: int = 900,
        max_injected_chars_total: int = 900,
        candidate_count: int = 12,
        pair_buffer_size: int = 64,
        rfm_update_interval: int = 5,
        min_pair_count_for_rfm: int = 4,
        ridge_lambda: float = 1e-2,
        kernel_gamma: float = 1.0,
        agop_smoothing: float = 0.25,
        agop_min_weight: float = 0.05,
        agop_max_weight: float = 8.0,
        need_memory_gate_enabled: bool = True,
        gate_risk_threshold: float = 0.34,
        gate_similarity_threshold: float = 0.50,
        disable_gate_until_memory_count: int = 8,
        min_utility: float = 0.0,
        min_rfm_score: float = -0.25,
        warmup_min_rfm_score: float = -1.0,
        novelty_threshold: float = 0.35,
        conflict_similarity_threshold: float = 0.80,
        existing_solution_similarity_threshold: float = 0.90,
        force_write_until_memory_count: int = 16,
        signature_warmup_count: int = 2,
        replace_only_if_old_harmful: bool = True,
        trim_explanation_before_action: bool = True,
        max_action_observation_chars: int = 260,
        memory_block_title: str = "Previous example. For the current task, do not explain. Start directly with the required action format.",
        rfm_debug_memory_profile_enabled: bool = False,
        use_evaluator_outcome_only: bool = False,
    ):
        super().__init__()
        assert top_k > 0
        assert max_memory_count > 0
        assert max_snippet_length > 0
        assert max_injected_chars_total > 0
        assert candidate_count >= top_k
        assert pair_buffer_size > 0
        assert rfm_update_interval > 0
        assert ridge_lambda > 0
        assert kernel_gamma > 0
        assert 0.0 <= agop_smoothing <= 1.0
        self.embedding_model_path = embedding_model_path
        self.embedding_device = embedding_device
        self.embedding_max_length = embedding_max_length
        self.embedding_cache_enabled = embedding_cache_enabled
        self.top_k = top_k
        self.max_memory_count = max_memory_count
        self.max_snippet_length = max_snippet_length
        self.max_injected_chars_total = max_injected_chars_total
        self.candidate_count = candidate_count
        self.pair_buffer_size = pair_buffer_size
        self.rfm_update_interval = rfm_update_interval
        self.min_pair_count_for_rfm = min_pair_count_for_rfm
        self.ridge_lambda = ridge_lambda
        self.kernel_gamma = kernel_gamma
        self.agop_smoothing = agop_smoothing
        self.agop_min_weight = agop_min_weight
        self.agop_max_weight = agop_max_weight
        self.need_memory_gate_enabled = need_memory_gate_enabled
        self.gate_risk_threshold = gate_risk_threshold
        self.gate_similarity_threshold = gate_similarity_threshold
        self.disable_gate_until_memory_count = disable_gate_until_memory_count
        self.min_utility = min_utility
        self.min_rfm_score = min_rfm_score
        self.warmup_min_rfm_score = warmup_min_rfm_score
        self.novelty_threshold = novelty_threshold
        self.conflict_similarity_threshold = conflict_similarity_threshold
        self.existing_solution_similarity_threshold = existing_solution_similarity_threshold
        self.force_write_until_memory_count = force_write_until_memory_count
        self.signature_warmup_count = signature_warmup_count
        self.replace_only_if_old_harmful = replace_only_if_old_harmful
        self.trim_explanation_before_action = trim_explanation_before_action
        self.max_action_observation_chars = max_action_observation_chars
        self.memory_block_title = memory_block_title
        self.rfm_debug_memory_profile_enabled = rfm_debug_memory_profile_enabled
        self.use_evaluator_outcome_only = use_evaluator_outcome_only

        self.memory_record_list: list[TokenWorthMemoryRecord] = []
        self.pair_record_list: list[TokenWorthPairRecord] = []
        self.embedding_cache_dict: dict[str, list[float]] = {}
        self.agop_diag_list: list[float] = []
        self.kernel_alpha_list: list[float] = []
        self.metrics = TokenWorthMetrics()
        self._next_memory_id = 0
        self._current_query = ""
        self._current_query_embedding: list[float] = []
        self._last_selected_id_list: list[int] = []
        self._last_selected_pair_feature_dict: dict[int, list[float]] = {}
        self._embedding_tokenizer = None
        self._embedding_model = None

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_memory_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_rfm_memory_records.json")

    def _get_pair_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_rfm_pair_records.json")

    def _get_embedding_cache_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_embedding_cache.json")

    def _get_agop_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_agop_diag.json")

    def _get_alpha_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_kernel_alpha.json")

    def _get_metrics_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_metrics.json")

    def _get_rfm_debug_memory_profile_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_rfm_update_memory_profile.jsonl")

    def restore_state(self) -> None:
        self.memory_record_list = load_json_state_list(
            self._get_memory_path(), TokenWorthMemoryRecord
        )
        self.pair_record_list = load_json_state_list(
            self._get_pair_path(), TokenWorthPairRecord
        )
        if os.path.exists(self._get_embedding_cache_path()):
            self.embedding_cache_dict = json.load(open(self._get_embedding_cache_path()))
        if os.path.exists(self._get_agop_path()):
            self.agop_diag_list = [
                float(value) for value in json.load(open(self._get_agop_path()))
            ]
        if os.path.exists(self._get_alpha_path()):
            self.kernel_alpha_list = [
                float(value) for value in json.load(open(self._get_alpha_path()))
            ]
        if os.path.exists(self._get_metrics_path()):
            self.metrics = TokenWorthMetrics.model_validate(
                json.load(open(self._get_metrics_path()))
            )
        if len(self.memory_record_list) > 0:
            self._next_memory_id = max(
                record.memory_id for record in self.memory_record_list
            ) + 1

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_memory_path(), self.memory_record_list)
        dump_json_state_list(self._get_pair_path(), self.pair_record_list)
        if self.embedding_cache_enabled:
            json.dump(self.embedding_cache_dict, open(self._get_embedding_cache_path(), "w"))
        json.dump(self.agop_diag_list, open(self._get_agop_path(), "w"), indent=2)
        json.dump(self.kernel_alpha_list, open(self._get_alpha_path(), "w"), indent=2)
        json.dump(self.metrics.model_dump(), open(self._get_metrics_path(), "w"), indent=2)

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        self.metrics.step += 1
        self._last_selected_id_list = []
        self._last_selected_pair_feature_dict = {}
        query = extract_question_from_session(callback_args.current_session).strip()
        if len(query) == 0:
            return
        self._current_query = query
        self._current_query_embedding = self._encode_text(query)
        if len(self.memory_record_list) == 0:
            self.metrics.skip_retrieve_count += 1
            return
        if self.need_memory_gate_enabled and not self._should_use_memory(query):
            self.metrics.gate_skip_count += 1
            self.metrics.skip_retrieve_count += 1
            return
        selected_record_list = self._select_memory_records(query)
        if len(selected_record_list) == 0:
            self.metrics.skip_retrieve_count += 1
            return
        self.metrics.retrieve_count += 1
        self._last_selected_id_list = [record.memory_id for record in selected_record_list]
        per_memory_chars = max(160, self.max_injected_chars_total // len(selected_record_list))
        self._inject_memory_into_instruction_slot(
            callback_args.current_session.chat_history,
            [record.render(per_memory_chars) for record in selected_record_list],
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        current_session = callback_args.current_session
        is_correct = self._is_successful_session(current_session)
        selected_record_list = [
            record
            for record in self.memory_record_list
            if record.memory_id in set(self._last_selected_id_list)
        ]
        for record in selected_record_list:
            self._update_memory_feedback(record, is_correct)
            pair_feature = self._last_selected_pair_feature_dict.get(record.memory_id)
            if pair_feature is None:
                pair_feature = self._build_pair_feature(
                    self._current_query_embedding, record
                )
            self._append_pair_record(pair_feature, 1.0 if is_correct else -1.0, record.memory_id)
        if len(selected_record_list) > 0 and len(self.pair_record_list) >= self.min_pair_count_for_rfm:
            if self.metrics.pair_write_count % self.rfm_update_interval == 0:
                self._update_rfm()
        if is_correct:
            new_record = self._build_memory_record_from_session(current_session)
            if new_record is not None:
                self._write_update_or_skip_memory(new_record, selected_record_list)
        self._evict_if_needed()

    def _is_successful_session(self, session: Session) -> bool:
        return (
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            and (
                self.use_evaluator_outcome_only
                or session.sample_status == SampleStatus.COMPLETED
            )
        )

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        # Execution normalization belongs to the shared agent protocol. These
        # helpers remain available for constructing compact memory snippets.
        pass

    def _should_use_memory(self, query: str) -> bool:
        if len(self.memory_record_list) < self.disable_gate_until_memory_count:
            return True
        risk_score = self._compute_gate_risk_score(query)
        if risk_score >= self.gate_risk_threshold:
            return True
        best_similarity = max(
            (
                self._cosine(self._current_query_embedding, record.query_embedding)
                for record in self.memory_record_list
                if record.utility >= self.min_utility
            ),
            default=0.0,
        )
        return best_similarity >= self.gate_similarity_threshold

    def _compute_gate_risk_score(self, query: str) -> float:
        token_list = tokenize(query)
        lowered = query.lower()
        length_score = min(1.0, len(token_list) / 90.0)
        constraint_score = min(
            1.0,
            sum(
                1
                for pattern in [
                    r"\band\b",
                    r"\bor\b",
                    r"\beither\b",
                    r"\bboth\b",
                    r"\bexcept\b",
                    r"\bunless\b",
                    r"\bbefore\b",
                    r"\bafter\b",
                    r"\bgreater than\b",
                    r"\bless than\b",
                    r"\bat least\b",
                    r"\bmore than\b",
                ]
                if re.search(pattern, lowered)
            )
            / 6.0,
        )
        multi_step_score = 1.0 if re.search(
            r"\bthen\b|\bafter\b|\bbefore\b|\bfirst\b|\bfinally\b|\bbased on\b|\bcopying\b",
            lowered,
        ) else 0.0
        state_change_score = 1.0 if re.search(
            r"\binsert\b|\bupdate\b|\bdelete\b|\bremove\b|\badd\b|\bcreate\b|\bmodify\b|\bchange\b|\bwrite\b|\bsave\b|\bmove\b|\bcopy\b|\breplace\b",
            lowered,
        ) else 0.0
        boundary_score = min(
            1.0,
            sum(
                1
                for pattern in [
                    r"\bhighest\b",
                    r"\blowest\b",
                    r"\bmaximum\b",
                    r"\bminimum\b",
                    r"\baverage\b",
                    r"\btop\b",
                    r"\blimit\b",
                    r"\boffset\b",
                    r"\bduplicate\b",
                    r"\bconflict\b",
                    r"\bnull\b",
                    r"\bempty\b",
                ]
                if re.search(pattern, lowered)
            )
            / 4.0,
        )
        return (
            0.22 * length_score
            + 0.24 * constraint_score
            + 0.20 * multi_step_score
            + 0.18 * state_change_score
            + 0.16 * boundary_score
        )

    def _select_memory_records(self, query: str) -> list[TokenWorthMemoryRecord]:
        candidate_record_list = [
            record for record in self.memory_record_list if record.utility >= self.min_utility
        ]
        if len(candidate_record_list) == 0:
            return []
        candidate_index_list = rank_texts_by_relevance(
            query,
            [record.search_text for record in candidate_record_list],
            min(self.candidate_count, len(candidate_record_list)),
        )
        candidate_record_list = [candidate_record_list[index] for index in candidate_index_list]
        scored_record_list: list[tuple[float, TokenWorthMemoryRecord, list[float]]] = []
        for record in candidate_record_list:
            pair_feature = self._build_pair_feature(self._current_query_embedding, record)
            score = self._predict_pair_score(pair_feature)
            if len(self.kernel_alpha_list) == 0:
                score += 0.05 * record.utility + 0.05 * record.reliability()
            scored_record_list.append((score, record, pair_feature))
        scored_record_list.sort(key=lambda item: (item[0], -item[1].memory_id), reverse=True)
        selected_record_list: list[TokenWorthMemoryRecord] = []
        score_threshold = (
            self.warmup_min_rfm_score
            if len(self.kernel_alpha_list) == 0
            or len(self.pair_record_list) < self.min_pair_count_for_rfm
            else self.min_rfm_score
        )
        for score, record, pair_feature in scored_record_list:
            if score < score_threshold:
                continue
            selected_record_list.append(record)
            self._last_selected_pair_feature_dict[record.memory_id] = pair_feature
            record.access_count += 1
            record.last_access_step = self.metrics.step
            if len(selected_record_list) >= self.top_k:
                break
        return selected_record_list

    def _build_pair_feature(
        self,
        query_embedding: Sequence[float],
        record: TokenWorthMemoryRecord,
    ) -> list[float]:
        memory_embedding = record.query_embedding
        difference = [abs(q - m) for q, m in zip(query_embedding, memory_embedding)]
        recency_gap = max(0, self.metrics.step - record.last_access_step)
        scalar_feature_list = [
            self._cosine(query_embedding, memory_embedding),
            max(-2.0, min(5.0, record.utility)) / 5.0,
            record.reliability(),
            math.exp(-0.02 * recency_gap),
        ]
        feature = difference + scalar_feature_list
        self._ensure_agop_diag(len(feature))
        return feature

    def _predict_pair_score(self, pair_feature: Sequence[float]) -> float:
        if len(self.kernel_alpha_list) == 0 or len(self.pair_record_list) == 0:
            if len(pair_feature) < 4:
                return 0.0
            cosine_score = pair_feature[-4]
            utility_score = pair_feature[-3]
            reliability_score = pair_feature[-2]
            recency_score = pair_feature[-1]
            return float(
                cosine_score
                + 0.05 * utility_score
                + 0.05 * reliability_score
                + 0.02 * recency_score
            )
        return float(
            sum(
                alpha * self._kernel(record.pair_feature, pair_feature)
                for alpha, record in zip(self.kernel_alpha_list, self.pair_record_list)
            )
        )

    def _append_pair_record(self, pair_feature: list[float], label: float, memory_id: int) -> None:
        self.pair_record_list.append(
            TokenWorthPairRecord(
                pair_feature=pair_feature,
                label=label,
                memory_id=memory_id,
                created_step=self.metrics.step,
            )
        )
        if label > 0:
            self.metrics.positive_pair_count += 1
        else:
            self.metrics.negative_pair_count += 1
        self.metrics.pair_write_count += 1
        if len(self.pair_record_list) > self.pair_buffer_size:
            self.pair_record_list = self.pair_record_list[-self.pair_buffer_size :]
            if len(self.kernel_alpha_list) != len(self.pair_record_list):
                self.kernel_alpha_list = []

    def _update_rfm(self) -> None:
        try:
            import numpy as np
        except Exception:
            return
        if len(self.pair_record_list) < self.min_pair_count_for_rfm:
            return
        debug_record: dict[str, object] | None = None
        if self.rfm_debug_memory_profile_enabled:
            debug_record = {
                "step": self.metrics.step,
                "pair_write_count": self.metrics.pair_write_count,
                "rfm_update_count_before": self.metrics.rfm_update_count,
                "pair_count": len(self.pair_record_list),
                "rss_mib": [],
            }

        def mark(stage: str) -> None:
            if debug_record is None:
                return
            debug_record["rss_mib"].append(
                {
                    "stage": stage,
                    "rss_mib": self._get_current_rss_mib(),
                }
            )

        mark("enter")
        z_matrix = np.asarray(
            [record.pair_feature for record in self.pair_record_list],
            dtype=float,
        )
        mark("after_z_matrix")
        y_vector = np.asarray([record.label for record in self.pair_record_list], dtype=float)
        mark("after_y_vector")
        self._ensure_agop_diag(z_matrix.shape[1])
        metric_diag = np.asarray(self.agop_diag_list, dtype=float)
        mark("after_metric_diag")
        diff = z_matrix[:, None, :] - z_matrix[None, :, :]
        mark("after_diff")
        distance = np.sqrt(np.maximum(np.sum(metric_diag * diff * diff, axis=2), 1e-12))
        mark("after_distance")
        kernel_matrix = np.exp(-self.kernel_gamma * distance)
        mark("after_kernel_matrix")
        regularized = kernel_matrix + self.ridge_lambda * np.eye(len(z_matrix))
        mark("after_regularized")
        try:
            alpha = np.linalg.solve(regularized, y_vector)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(regularized, y_vector, rcond=None)[0]
        mark("after_solve")
        gradient_matrix = []
        for point in z_matrix:
            point_diff = point[None, :] - z_matrix
            point_distance = np.sqrt(
                np.maximum(np.sum(metric_diag * point_diff * point_diff, axis=1), 1e-12)
            )
            kernel_vector = np.exp(-self.kernel_gamma * point_distance)
            coeff = -self.kernel_gamma * alpha * kernel_vector / point_distance
            gradient = np.sum(coeff[:, None] * metric_diag * point_diff, axis=0)
            gradient_matrix.append(gradient)
        mark("after_gradient_loop")
        gradient_matrix = np.asarray(gradient_matrix, dtype=float)
        mark("after_gradient_matrix")
        agop_diag = np.mean(gradient_matrix * gradient_matrix, axis=0)
        mark("after_agop_diag")
        agop_mean = float(np.mean(agop_diag))
        if agop_mean > 1e-12:
            agop_diag = agop_diag / agop_mean
        updated_diag = (1.0 - self.agop_smoothing) * metric_diag + self.agop_smoothing * agop_diag
        updated_diag = np.clip(updated_diag, self.agop_min_weight, self.agop_max_weight)
        updated_diag = updated_diag / max(float(np.mean(updated_diag)), 1e-12)
        self.agop_diag_list = updated_diag.astype(float).tolist()
        self.kernel_alpha_list = alpha.astype(float).tolist()
        self.metrics.rfm_update_count += 1
        mark("exit")
        if debug_record is not None:
            debug_record["rfm_update_count_after"] = self.metrics.rfm_update_count
            debug_record["feature_dim"] = int(z_matrix.shape[1])
            with open(
                self._get_rfm_debug_memory_profile_path(),
                "a",
                encoding="utf-8",
            ) as file:
                file.write(json.dumps(debug_record) + "\n")

    def _get_current_rss_mib(self) -> float:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as file:
                resident_pages = int(file.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
        except Exception:
            return -1.0

    def _kernel(self, left_feature: Sequence[float], right_feature: Sequence[float]) -> float:
        self._ensure_agop_diag(len(left_feature))
        weighted_distance_square = sum(
            weight * (left - right) * (left - right)
            for weight, left, right in zip(self.agop_diag_list, left_feature, right_feature)
        )
        return math.exp(-self.kernel_gamma * math.sqrt(max(weighted_distance_square, 1e-12)))

    def _ensure_agop_diag(self, dimension: int) -> None:
        if len(self.agop_diag_list) == dimension:
            return
        self.agop_diag_list = [1.0 for _ in range(dimension)]
        self.kernel_alpha_list = []

    def _build_memory_record_from_session(
        self, session: Session
    ) -> TokenWorthMemoryRecord | None:
        query = extract_question_from_session(session).strip()
        if len(query) == 0:
            return None
        trajectory = self._build_action_only_trajectory(session)
        if len(trajectory) == 0:
            return None
        snippet = truncate_text(f"Question {query}:\n{trajectory}", self.max_snippet_length)
        embedding = self._encode_text(query)
        return TokenWorthMemoryRecord(
            memory_id=self._next_memory_id,
            query=query,
            snippet=snippet,
            search_text=f"{query}\n{trajectory}\n{self._build_signature(query)}",
            query_embedding=embedding,
            signature=self._build_signature(query),
            created_step=self.metrics.step,
            last_access_step=self.metrics.step,
        )

    def _build_action_only_trajectory(self, session: Session) -> str:
        item_text_list: list[str] = []
        for item_index in range(3, session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(item_index)
            if item.role == Role.AGENT:
                action_text = self._extract_action_text(item.content)
                if len(action_text) > 0:
                    item_text_list.append(f"assistant: {action_text}")
            elif item.role == Role.USER:
                observation = truncate_text(
                    " ".join(item.content.split()),
                    self.max_action_observation_chars,
                )
                if len(observation) > 0:
                    item_text_list.append(f"user: {observation}")
        return "\n".join(item_text_list)

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        most_similar_record, similarity = self._find_most_similar_memory(new_record)
        if len(self.memory_record_list) < self.force_write_until_memory_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            return
        same_signature_count = sum(
            1
            for record in self.memory_record_list
            if record.signature == new_record.signature
        )
        if same_signature_count < self.signature_warmup_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            return
        if (
            len(selected_record_list) > 0
            and most_similar_record is not None
            and most_similar_record.memory_id in {record.memory_id for record in selected_record_list}
            and similarity >= self.existing_solution_similarity_threshold
            and most_similar_record.utility >= 0.5
        ):
            self.metrics.solved_by_existing_memory_count += 1
            return
        old_memory_is_harmful = (
            most_similar_record is not None
            and (
                most_similar_record.utility < 0.0
                or most_similar_record.harm_count > most_similar_record.help_count
            )
        )
        if (
            most_similar_record is not None
            and similarity >= self.conflict_similarity_threshold
            and most_similar_record.signature == new_record.signature
            and most_similar_record.snippet != new_record.snippet
            and most_similar_record.utility <= new_record.utility
            and (not self.replace_only_if_old_harmful or old_memory_is_harmful)
        ):
            replace_index = self.memory_record_list.index(most_similar_record)
            new_record.memory_id = most_similar_record.memory_id
            self.memory_record_list[replace_index] = new_record
            self.metrics.replace_count += 1
            self.metrics.conflict_replace_count += 1
            return
        if similarity < self.novelty_threshold or most_similar_record is None:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            return
        if most_similar_record.utility < 0.5:
            replace_index = self.memory_record_list.index(most_similar_record)
            new_record.memory_id = most_similar_record.memory_id
            self.memory_record_list[replace_index] = new_record
            self.metrics.replace_count += 1
            return

    def _append_memory(self, record: TokenWorthMemoryRecord) -> None:
        record.memory_id = self._next_memory_id
        self._next_memory_id += 1
        self.memory_record_list.append(record)
        self.metrics.write_count += 1

    def _find_most_similar_memory(
        self, new_record: TokenWorthMemoryRecord
    ) -> tuple[TokenWorthMemoryRecord | None, float]:
        if len(self.memory_record_list) == 0:
            return None, 0.0
        scored_record_list = [
            (self._cosine(new_record.query_embedding, record.query_embedding), record)
            for record in self.memory_record_list
        ]
        scored_record_list.sort(key=lambda item: item[0], reverse=True)
        return scored_record_list[0][1], scored_record_list[0][0]

    def _update_memory_feedback(self, record: TokenWorthMemoryRecord, is_correct: bool) -> None:
        if is_correct:
            record.help_count += 1
            record.utility = min(5.0, record.utility + 0.25)
        else:
            record.harm_count += 1
            record.utility = max(-2.0, record.utility - 0.45)

    def _evict_if_needed(self) -> None:
        if len(self.memory_record_list) <= self.max_memory_count:
            return
        self.memory_record_list.sort(
            key=lambda record: (
                record.utility,
                record.help_count - record.harm_count,
                record.access_count,
                record.last_access_step,
            ),
            reverse=True,
        )
        evicted_count = len(self.memory_record_list) - self.max_memory_count
        self.memory_record_list = self.memory_record_list[: self.max_memory_count]
        self.metrics.evict_count += evicted_count

    def _inject_memory_into_instruction_slot(
        self,
        chat_history: ChatHistory,
        memory_text_list: Sequence[str],
    ) -> None:
        if len(memory_text_list) == 0:
            return
        original_first_item = chat_history.get_item_deep_copy(0)
        assert original_first_item.role == Role.USER
        memory_block = "\n".join(memory_text_list)
        if len(self.memory_block_title.strip()) > 0:
            memory_block = f"{self.memory_block_title}\n{memory_block}"
        anchor = "\n\nNow, I will give you the question that you need to solve."
        if anchor in original_first_item.content:
            updated_content = original_first_item.content.replace(
                anchor, f"\n\n{memory_block}{anchor}", 1
            )
        else:
            updated_content = f"{original_first_item.content}\n\n{memory_block}"
        chat_history.set(0, ChatHistoryItem(role=Role.USER, content=updated_content))

    def _remove_counterfeit_user_response(self, content: str) -> str:
        location = content.find("\nuser: ")
        if location != -1:
            return content[:location]
        return content

    def _extract_action_text(self, content: str) -> str:
        content = self._remove_counterfeit_user_response(content)
        earliest_match = None
        for pattern in _ACTION_PATTERN_LIST:
            match = pattern.search(content)
            if match is None:
                continue
            if earliest_match is None or match.start() < earliest_match.start():
                earliest_match = match
        if earliest_match is None:
            return ""
        return earliest_match.group(0).strip()

    def _build_signature(self, text: str) -> str:
        lowered = text.lower()
        signature_token_list: list[str] = []
        for name, pattern in {
            "read": r"\bselect\b|\breturn\b|\bfind\b|\bshow\b|\bwhat\b|\blist\b",
            "create": r"\binsert\b|\badd\b|\bcreate\b|\bnew\b",
            "update": r"\bupdate\b|\bchange\b|\bmodify\b|\bset\b|\breplace\b",
            "delete": r"\bdelete\b|\bremove\b",
            "multi": r"\band\b|\bor\b|\bthen\b|\bafter\b|\bbefore\b",
            "aggregate": r"\bcount\b|\bsum\b|\bavg\b|\baverage\b|\btotal\b|\bmaximum\b|\bminimum\b",
            "order": r"\border\b|\btop\b|\blimit\b|\boffset\b|\bhighest\b|\blowest\b",
            "tool": r"\bbash\b|\bshell\b|\bcommand\b|\bfile\b|\bdirectory\b",
        }.items():
            if re.search(pattern, lowered):
                signature_token_list.append(name)
        if len(signature_token_list) == 0:
            signature_token_list.append("generic")
        return ",".join(signature_token_list)

    def _encode_text(self, text: str) -> list[float]:
        cache_key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        if self.embedding_cache_enabled and cache_key in self.embedding_cache_dict:
            return self.embedding_cache_dict[cache_key]
        vector = self._encode_dense_vector(text)
        if self.embedding_cache_enabled:
            self.embedding_cache_dict[cache_key] = vector
        return vector

    def _ensure_embedding_model_loaded(self) -> None:
        if self._embedding_model is not None and self._embedding_tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._embedding_tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_path, local_files_only=True
        )
        self._embedding_model = AutoModel.from_pretrained(
            self.embedding_model_path, local_files_only=True
        )
        self._embedding_model.to(self.embedding_device)
        self._embedding_model.eval()

    def _encode_dense_vector(self, text: str) -> list[float]:
        self._ensure_embedding_model_loaded()
        import torch
        import torch.nn.functional as functional

        assert self._embedding_tokenizer is not None
        assert self._embedding_model is not None
        encoded_input = self._embedding_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.embedding_max_length,
            return_tensors="pt",
        )
        encoded_input = {
            key: value.to(self.embedding_device) for key, value in encoded_input.items()
        }
        with torch.no_grad():
            model_output = self._embedding_model(**encoded_input)
        token_embedding = model_output.last_hidden_state
        attention_mask = encoded_input["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embedding * attention_mask).sum(dim=1) / attention_mask.sum(
            dim=1
        ).clamp(min=1e-9)
        pooled = functional.normalize(pooled, p=2, dim=1)
        return pooled[0].detach().cpu().tolist()

    def _cosine(self, left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) == 0 or len(right) == 0:
            return 0.0
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return 0.0
        return dot / (left_norm * right_norm)


class TokenWorthFullCandidateMemoryCallback(TokenWorthBaseMemoryCallback):
    """TokenWorth retrieval variant that scores every memory candidate.

    This keeps the DB-specific signatures, warmup, RFM scoring, memory writing,
    and prompt injection behavior from TokenWorthBaseMemoryCallback. The only intended
    change is that all utility-qualified memories are scored by RFM instead of
    first being pruned to a BM25 candidate set.
    """

    def _select_memory_records(self, query: str) -> list[TokenWorthMemoryRecord]:
        candidate_record_list = [
            record
            for record in self.memory_record_list
            if record.utility >= self.min_utility
        ]
        if len(candidate_record_list) == 0:
            return []

        scored_record_list: list[tuple[float, TokenWorthMemoryRecord, list[float]]] = []
        for record in candidate_record_list:
            pair_feature = self._build_pair_feature(
                self._current_query_embedding, record
            )
            score = self._predict_pair_score(pair_feature)
            if len(self.kernel_alpha_list) == 0:
                score += 0.05 * record.utility + 0.05 * record.reliability()
            scored_record_list.append((score, record, pair_feature))
        scored_record_list.sort(
            key=lambda item: (item[0], -item[1].memory_id),
            reverse=True,
        )

        score_threshold = (
            self.warmup_min_rfm_score
            if len(self.kernel_alpha_list) == 0
            or len(self.pair_record_list) < self.min_pair_count_for_rfm
            else self.min_rfm_score
        )
        selected_record_list: list[TokenWorthMemoryRecord] = []
        for score, record, pair_feature in scored_record_list:
            if score < score_threshold:
                continue
            selected_record_list.append(record)
            self._last_selected_pair_feature_dict[record.memory_id] = pair_feature
            record.access_count += 1
            record.last_access_step = self.metrics.step
            if len(selected_record_list) >= self.top_k:
                break
        return selected_record_list


class TokenWorthAlwaysRetrieveMemoryCallback(TokenWorthFullCandidateMemoryCallback):
    """TokenWorth retrieval variant that always attempts retrieval.

    This variant keeps the DB-specific signatures, warmup memory writing,
    utility/reliability feedback, RFM/AGOP scoring, and prompt injection format.
    It removes two retrieval-side controls:

    - BM25 candidate pruning: all utility-qualified memories are scored by RFM.
    - need-memory gate: retrieval is attempted whenever the memory bank is non-empty.
    """

    def __init__(self, *args, **kwargs):
        kwargs["need_memory_gate_enabled"] = False
        super().__init__(*args, **kwargs)

    def _should_use_memory(self, query: str) -> bool:
        return True


class TokenWorthFrequencyFeatureMemoryCallback(
    TokenWorthAlwaysRetrieveMemoryCallback
):
    """TokenWorth retrieval variant with a memory access-frequency feature."""

    def _build_pair_feature(
        self,
        query_embedding: Sequence[float],
        record: TokenWorthMemoryRecord,
    ) -> list[float]:
        memory_embedding = record.query_embedding
        difference = [abs(q - m) for q, m in zip(query_embedding, memory_embedding)]
        recency_gap = max(0, self.metrics.step - record.last_access_step)
        memory_age = max(1, self.metrics.step - record.created_step + 1)
        access_frequency = min(1.0, record.access_count / memory_age)
        scalar_feature_list = [
            self._cosine(query_embedding, memory_embedding),
            max(-2.0, min(5.0, record.utility)) / 5.0,
            record.reliability(),
            access_frequency,
            math.exp(-0.02 * recency_gap),
        ]
        feature = difference + scalar_feature_list
        self._ensure_agop_diag(len(feature))
        return feature

    def _predict_pair_score(self, pair_feature: Sequence[float]) -> float:
        if len(self.kernel_alpha_list) == 0 or len(self.pair_record_list) == 0:
            if len(pair_feature) < 5:
                return 0.0
            cosine_score = pair_feature[-5]
            utility_score = pair_feature[-4]
            reliability_score = pair_feature[-3]
            frequency_score = pair_feature[-2]
            recency_score = pair_feature[-1]
            return float(
                cosine_score
                + 0.05 * utility_score
                + 0.05 * reliability_score
                + 0.02 * frequency_score
                + 0.02 * recency_score
            )
        return float(
            sum(
                alpha * self._kernel(record.pair_feature, pair_feature)
                for alpha, record in zip(
                    self.kernel_alpha_list, self.pair_record_list
                )
            )
        )


class TokenWorthFeedbackBankMemoryCallback(
    TokenWorthFrequencyFeatureMemoryCallback
):
    """TokenWorth feedback-aware memory-bank controller.

    Adds RFM confidence checks, recent-query probes, conservative write gain,
    protected eviction, and explicit management metrics.
    """

    def __init__(
        self,
        *args,
        same_signature_probe_count: int = 4,
        recent_probe_count: int = 8,
        rfm_management_min_pair_count: int = 16,
        rfm_management_min_update_count: int = 3,
        rfm_management_min_positive_count: int = 4,
        rfm_management_min_negative_count: int = 4,
        rfm_positive_ratio_threshold: float = 0.60,
        rfm_win_ratio_threshold: float = 0.60,
        rfm_write_gain_margin: float = 0.08,
        rfm_replace_margin: float = 0.05,
        rfm_skip_margin: float = 0.02,
        embedding_novelty_threshold: float = 0.60,
        protected_min_help_count: int = 4,
        protected_min_utility: float = 1.2,
        eviction_rfm_weight: float = 0.20,
        eviction_utility_weight: float = 0.40,
        eviction_reliability_weight: float = 0.25,
        eviction_access_weight: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.same_signature_probe_count = same_signature_probe_count
        self.recent_probe_count = recent_probe_count
        self.rfm_management_min_pair_count = rfm_management_min_pair_count
        self.rfm_management_min_update_count = rfm_management_min_update_count
        self.rfm_management_min_positive_count = rfm_management_min_positive_count
        self.rfm_management_min_negative_count = rfm_management_min_negative_count
        self.rfm_positive_ratio_threshold = rfm_positive_ratio_threshold
        self.rfm_win_ratio_threshold = rfm_win_ratio_threshold
        self.rfm_write_gain_margin = rfm_write_gain_margin
        self.rfm_replace_margin = rfm_replace_margin
        self.rfm_skip_margin = rfm_skip_margin
        self.embedding_novelty_threshold = embedding_novelty_threshold
        self.protected_min_help_count = protected_min_help_count
        self.protected_min_utility = protected_min_utility
        self.eviction_rfm_weight = eviction_rfm_weight
        self.eviction_utility_weight = eviction_utility_weight
        self.eviction_reliability_weight = eviction_reliability_weight
        self.eviction_access_weight = eviction_access_weight

        self.recent_query_embedding_list: list[list[float]] = []
        self.bank_management_metrics: dict[str, int] = {
            "rfm_ready_count": 0,
            "rfm_not_ready_fallback_count": 0,
            "warmup_write_count": 0,
            "signature_warmup_write_count": 0,
            "rfm_write_accept_count": 0,
            "rfm_write_skip_count": 0,
            "rfm_novelty_write_count": 0,
            "rfm_conflict_replace_count": 0,
            "rfm_conflict_skip_count": 0,
            "rfm_existing_replace_count": 0,
            "rfm_existing_skip_count": 0,
            "rfm_low_quality_replace_count": 0,
            "rfm_evict_count": 0,
            "rfm_evict_protected_count": 0,
        }

    def _get_recent_probe_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_recent_probe_embeddings.json")

    def _get_management_metrics_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_bank_management_metrics.json")

    def restore_state(self) -> None:
        super().restore_state()
        if os.path.exists(self._get_recent_probe_path()):
            self.recent_query_embedding_list = json.load(
                open(self._get_recent_probe_path())
            )
        if os.path.exists(self._get_management_metrics_path()):
            loaded_metrics = json.load(open(self._get_management_metrics_path()))
            self.bank_management_metrics.update(
                {key: int(value) for key, value in loaded_metrics.items()}
            )

    def on_state_save(self, callback_args) -> None:
        super().on_state_save(callback_args)
        json.dump(
            self.recent_query_embedding_list,
            open(self._get_recent_probe_path(), "w"),
        )
        json.dump(
            self.bank_management_metrics,
            open(self._get_management_metrics_path(), "w"),
            indent=2,
        )

    def on_task_complete(self, callback_args) -> None:
        if len(self._current_query_embedding) > 0 and len(self._last_selected_id_list) > 0:
            self.recent_query_embedding_list.append(self._current_query_embedding)
            self.recent_query_embedding_list = self.recent_query_embedding_list[
                -self.recent_probe_count :
            ]
        super().on_task_complete(callback_args)

    def _increment_management_metric(self, name: str, amount: int = 1) -> None:
        self.bank_management_metrics[name] = (
            self.bank_management_metrics.get(name, 0) + amount
        )

    def _rfm_management_ready(self, *, count_metric: bool = True) -> bool:
        ready = (
            len(self.kernel_alpha_list) > 0
            and len(self.pair_record_list) >= self.rfm_management_min_pair_count
            and self.metrics.rfm_update_count >= self.rfm_management_min_update_count
            and self.metrics.positive_pair_count
            >= self.rfm_management_min_positive_count
            and self.metrics.negative_pair_count
            >= self.rfm_management_min_negative_count
        )
        if ready and count_metric:
            self._increment_management_metric("rfm_ready_count")
        return ready

    def _build_probe_embedding_list(
        self,
        new_record: TokenWorthMemoryRecord,
    ) -> list[list[float]]:
        probe_embedding_list = [new_record.query_embedding]
        same_signature_record_list = [
            record
            for record in self.memory_record_list
            if record.signature == new_record.signature
        ]
        same_signature_record_list.sort(
            key=lambda record: (
                record.last_access_step,
                record.utility,
                record.help_count - record.harm_count,
            ),
            reverse=True,
        )
        for record in same_signature_record_list[: self.same_signature_probe_count]:
            probe_embedding_list.append(record.query_embedding)
        probe_embedding_list.extend(self.recent_query_embedding_list[-self.recent_probe_count :])
        return probe_embedding_list

    def _score_list_on_probe_embeddings(
        self,
        record: TokenWorthMemoryRecord,
        probe_embedding_list: Sequence[Sequence[float]],
    ) -> list[float]:
        return [
            self._predict_pair_score(self._build_pair_feature(probe_embedding, record))
            for probe_embedding in probe_embedding_list
        ]

    def _score_stats(self, score_list: Sequence[float]) -> dict[str, float]:
        if len(score_list) == 0:
            return {"mean": 0.0, "positive_ratio": 0.0, "std": 0.0}
        mean_score = sum(score_list) / len(score_list)
        positive_ratio = sum(1 for score in score_list if score > 0.0) / len(score_list)
        std_score = statistics.pstdev(score_list) if len(score_list) > 1 else 0.0
        return {
            "mean": mean_score,
            "positive_ratio": positive_ratio,
            "std": std_score,
        }

    def _win_ratio(
        self,
        new_score_list: Sequence[float],
        old_score_list: Sequence[float],
        margin: float,
    ) -> float:
        if len(new_score_list) == 0 or len(new_score_list) != len(old_score_list):
            return 0.0
        return (
            sum(
                1
                for new_score, old_score in zip(new_score_list, old_score_list)
                if new_score > old_score + margin
            )
            / len(new_score_list)
        )

    def _best_existing_score_stats(
        self,
        probe_embedding_list: Sequence[Sequence[float]],
        signature: str,
    ) -> dict[str, float]:
        best_stats = {"mean": float("-inf"), "positive_ratio": 0.0, "std": 0.0}
        for record in self.memory_record_list:
            if record.signature != signature or record.utility < self.min_utility:
                continue
            stats = self._score_stats(
                self._score_list_on_probe_embeddings(record, probe_embedding_list)
            )
            if stats["mean"] > best_stats["mean"]:
                best_stats = stats
        return best_stats

    def _max_query_similarity(self, new_record: TokenWorthMemoryRecord) -> float:
        if len(self.memory_record_list) == 0:
            return 0.0
        return max(
            self._cosine(new_record.query_embedding, record.query_embedding)
            for record in self.memory_record_list
        )

    def _replace_memory(
        self,
        old_record: TokenWorthMemoryRecord,
        new_record: TokenWorthMemoryRecord,
    ) -> None:
        replace_index = self.memory_record_list.index(old_record)
        new_record.memory_id = old_record.memory_id
        self.memory_record_list[replace_index] = new_record
        self.metrics.replace_count += 1

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        most_similar_record, similarity = self._find_most_similar_memory(new_record)

        if len(self.memory_record_list) < self.force_write_until_memory_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("warmup_write_count")
            return

        same_signature_count = sum(
            1
            for record in self.memory_record_list
            if record.signature == new_record.signature
        )
        if same_signature_count < self.signature_warmup_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("signature_warmup_write_count")
            return

        if not self._rfm_management_ready():
            self._increment_management_metric("rfm_not_ready_fallback_count")
            super()._write_update_or_skip_memory(new_record, selected_record_list)
            return

        probe_embedding_list = self._build_probe_embedding_list(new_record)
        new_score_list = self._score_list_on_probe_embeddings(
            new_record, probe_embedding_list
        )
        new_stats = self._score_stats(new_score_list)
        old_memory_is_harmful = (
            most_similar_record is not None
            and (
                most_similar_record.utility < 0.0
                or most_similar_record.harm_count > most_similar_record.help_count
            )
        )

        if (
            most_similar_record is not None
            and similarity >= self.conflict_similarity_threshold
            and most_similar_record.signature == new_record.signature
            and most_similar_record.snippet != new_record.snippet
        ):
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            win_ratio = self._win_ratio(
                new_score_list, old_score_list, self.rfm_replace_margin
            )
            if (
                new_stats["mean"] > old_stats["mean"] + self.rfm_replace_margin
                and win_ratio >= self.rfm_win_ratio_threshold
            ) or (
                old_memory_is_harmful
                and new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin
            ):
                self._replace_memory(most_similar_record, new_record)
                self.metrics.conflict_replace_count += 1
                self._increment_management_metric("rfm_conflict_replace_count")
            else:
                self._increment_management_metric("rfm_conflict_skip_count")
            return

        if (
            len(selected_record_list) > 0
            and most_similar_record is not None
            and most_similar_record.memory_id
            in {record.memory_id for record in selected_record_list}
            and similarity >= self.existing_solution_similarity_threshold
            and most_similar_record.utility >= 0.5
        ):
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            win_ratio = self._win_ratio(
                new_score_list, old_score_list, self.rfm_replace_margin
            )
            if (
                new_stats["mean"] > old_stats["mean"] + self.rfm_replace_margin
                and win_ratio >= self.rfm_win_ratio_threshold
            ):
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_existing_replace_count")
            else:
                self.metrics.solved_by_existing_memory_count += 1
                self._increment_management_metric("rfm_existing_skip_count")
            return

        best_old_stats = self._best_existing_score_stats(
            probe_embedding_list, new_record.signature
        )
        gain = new_stats["mean"] - best_old_stats["mean"]
        if (
            gain > self.rfm_write_gain_margin
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        if self._max_query_similarity(new_record) < self.embedding_novelty_threshold:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_novelty_write_count")
            return

        if most_similar_record is not None and most_similar_record.utility < 0.5:
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            win_ratio = self._win_ratio(
                new_score_list, old_score_list, self.rfm_replace_margin
            )
            if (
                new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin
                and win_ratio >= self.rfm_win_ratio_threshold
            ):
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_low_quality_replace_count")
                return

        self._increment_management_metric("rfm_write_skip_count")

    def _eviction_probe_embedding_list(self) -> list[list[float]]:
        probe_embedding_list = list(self.recent_query_embedding_list[-self.recent_probe_count :])
        record_list = sorted(
            self.memory_record_list,
            key=lambda record: (
                record.last_access_step,
                record.utility,
                record.help_count - record.harm_count,
            ),
            reverse=True,
        )
        for record in record_list[: max(1, self.same_signature_probe_count * 2)]:
            probe_embedding_list.append(record.query_embedding)
        return probe_embedding_list

    def _memory_is_protected(self, record: TokenWorthMemoryRecord) -> bool:
        return (
            record.help_count >= self.protected_min_help_count
            and record.utility >= self.protected_min_utility
        )

    def _memory_eviction_score(
        self,
        record: TokenWorthMemoryRecord,
        probe_embedding_list: Sequence[Sequence[float]],
    ) -> float:
        if self._rfm_management_ready(count_metric=False) and len(probe_embedding_list) > 0:
            rfm_score = self._score_stats(
                self._score_list_on_probe_embeddings(record, probe_embedding_list)
            )["mean"]
        else:
            rfm_score = 0.0
        utility_score = max(-2.0, min(5.0, record.utility)) / 5.0
        reliability_score = record.reliability()
        access_score = min(1.0, record.access_count / max(1, self.metrics.step))
        return (
            self.eviction_rfm_weight * rfm_score
            + self.eviction_utility_weight * utility_score
            + self.eviction_reliability_weight * reliability_score
            + self.eviction_access_weight * access_score
        )

    def _evict_if_needed(self) -> None:
        if len(self.memory_record_list) <= self.max_memory_count:
            return
        probe_embedding_list = self._eviction_probe_embedding_list()
        protected_record_list = [
            record for record in self.memory_record_list if self._memory_is_protected(record)
        ]
        candidate_record_list = [
            record for record in self.memory_record_list if not self._memory_is_protected(record)
        ]
        self._increment_management_metric(
            "rfm_evict_protected_count",
            max(0, len(protected_record_list) + len(candidate_record_list) - self.max_memory_count),
        )
        if len(candidate_record_list) == 0:
            candidate_record_list = list(self.memory_record_list)

        evict_count = len(self.memory_record_list) - self.max_memory_count
        if len(candidate_record_list) < evict_count:
            candidate_record_list = list(self.memory_record_list)
        candidate_record_list.sort(
            key=lambda record: (
                self._memory_eviction_score(record, probe_embedding_list),
                record.utility,
                record.help_count - record.harm_count,
                record.access_count,
                record.last_access_step,
            )
        )
        evict_id_set = {record.memory_id for record in candidate_record_list[:evict_count]}
        self.memory_record_list = [
            record for record in self.memory_record_list if record.memory_id not in evict_id_set
        ]
        self.metrics.evict_count += evict_count
        self._increment_management_metric("rfm_evict_count", evict_count)


class TokenWorthCoverageBankMemoryCallback(TokenWorthFeedbackBankMemoryCallback):
    """TokenWorth memory-bank controller with coverage repair.

    After warmup, write, replacement, and eviction decisions use RFM probe
    scores with conservative coverage repair to keep enough usable memories.
    """

    def __init__(
        self,
        *args,
        rfm_management_min_pair_count: int = 24,
        rfm_management_min_update_count: int = 5,
        rfm_positive_ratio_threshold: float = 0.45,
        rfm_win_ratio_threshold: float = 0.50,
        rfm_write_gain_margin: float = 0.02,
        rfm_replace_margin: float = 0.03,
        rfm_skip_margin: float = 0.10,
        embedding_novelty_threshold: float = 0.45,
        min_active_same_signature_count: int = 2,
        rfm_same_signature_replace_margin: float = 0.01,
        protected_min_help_count: int = 5,
        protected_min_utility: float = 1.4,
        eviction_rfm_weight: float = 0.30,
        eviction_utility_weight: float = 0.40,
        eviction_reliability_weight: float = 0.20,
        eviction_access_weight: float = 0.10,
        **kwargs,
    ):
        super().__init__(
            *args,
            rfm_management_min_pair_count=rfm_management_min_pair_count,
            rfm_management_min_update_count=rfm_management_min_update_count,
            rfm_positive_ratio_threshold=rfm_positive_ratio_threshold,
            rfm_win_ratio_threshold=rfm_win_ratio_threshold,
            rfm_write_gain_margin=rfm_write_gain_margin,
            rfm_replace_margin=rfm_replace_margin,
            rfm_skip_margin=rfm_skip_margin,
            embedding_novelty_threshold=embedding_novelty_threshold,
            protected_min_help_count=protected_min_help_count,
            protected_min_utility=protected_min_utility,
            eviction_rfm_weight=eviction_rfm_weight,
            eviction_utility_weight=eviction_utility_weight,
            eviction_reliability_weight=eviction_reliability_weight,
            eviction_access_weight=eviction_access_weight,
            **kwargs,
        )
        self.min_active_same_signature_count = min_active_same_signature_count
        self.rfm_same_signature_replace_margin = rfm_same_signature_replace_margin
        self.bank_management_metrics.setdefault("rfm_coverage_repair_write_count", 0)
        self.bank_management_metrics.setdefault("rfm_same_signature_replace_count", 0)
        self.bank_management_metrics.setdefault("rfm_controller_skip_count", 0)

    def _same_signature_active_count(self, signature: str) -> int:
        return sum(
            1
            for record in self.memory_record_list
            if record.signature == signature and record.utility >= self.min_utility
        )

    def _worst_same_signature_record(
        self,
        signature: str,
        probe_embedding_list: Sequence[Sequence[float]],
    ) -> tuple[TokenWorthMemoryRecord | None, dict[str, float]]:
        worst_record = None
        worst_stats = {"mean": float("inf"), "positive_ratio": 0.0, "std": 0.0}
        for record in self.memory_record_list:
            if record.signature != signature:
                continue
            stats = self._score_stats(
                self._score_list_on_probe_embeddings(record, probe_embedding_list)
            )
            score_tuple = (
                stats["mean"],
                record.utility,
                record.reliability(),
                record.access_count,
            )
            if worst_record is None or score_tuple < (
                worst_stats["mean"],
                worst_record.utility,
                worst_record.reliability(),
                worst_record.access_count,
            ):
                worst_record = record
                worst_stats = stats
        return worst_record, worst_stats

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        most_similar_record, similarity = self._find_most_similar_memory(new_record)

        if len(self.memory_record_list) < self.force_write_until_memory_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("warmup_write_count")
            return

        same_signature_count = sum(
            1
            for record in self.memory_record_list
            if record.signature == new_record.signature
        )
        if same_signature_count < self.signature_warmup_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("signature_warmup_write_count")
            return

        if not self._rfm_management_ready():
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_coverage_repair_write_count")
            return

        probe_embedding_list = self._build_probe_embedding_list(new_record)
        new_score_list = self._score_list_on_probe_embeddings(
            new_record, probe_embedding_list
        )
        new_stats = self._score_stats(new_score_list)

        old_memory_is_harmful = (
            most_similar_record is not None
            and (
                most_similar_record.utility < 0.0
                or most_similar_record.harm_count > most_similar_record.help_count
            )
        )

        if (
            most_similar_record is not None
            and similarity >= self.conflict_similarity_threshold
            and most_similar_record.signature == new_record.signature
            and most_similar_record.snippet != new_record.snippet
        ):
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            win_ratio = self._win_ratio(
                new_score_list, old_score_list, self.rfm_replace_margin
            )
            if (
                new_stats["mean"] > old_stats["mean"] + self.rfm_replace_margin
                and win_ratio >= self.rfm_win_ratio_threshold
            ) or (
                old_memory_is_harmful
                and new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin
            ):
                self._replace_memory(most_similar_record, new_record)
                self.metrics.conflict_replace_count += 1
                self._increment_management_metric("rfm_conflict_replace_count")
            else:
                self._increment_management_metric("rfm_conflict_skip_count")
            return

        if (
            len(selected_record_list) > 0
            and most_similar_record is not None
            and most_similar_record.memory_id
            in {record.memory_id for record in selected_record_list}
            and similarity >= self.existing_solution_similarity_threshold
            and most_similar_record.utility >= 0.5
        ):
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            win_ratio = self._win_ratio(
                new_score_list, old_score_list, self.rfm_replace_margin
            )
            if (
                new_stats["mean"] > old_stats["mean"] + self.rfm_replace_margin
                and win_ratio >= self.rfm_win_ratio_threshold
            ):
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_existing_replace_count")
            else:
                self.metrics.solved_by_existing_memory_count += 1
                self._increment_management_metric("rfm_existing_skip_count")
            return

        if most_similar_record is None or similarity < self.novelty_threshold:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        best_old_stats = self._best_existing_score_stats(
            probe_embedding_list, new_record.signature
        )
        gain = new_stats["mean"] - best_old_stats["mean"]
        if (
            gain > self.rfm_write_gain_margin
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        active_same_signature_count = self._same_signature_active_count(
            new_record.signature
        )
        if (
            active_same_signature_count < self.min_active_same_signature_count
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_coverage_repair_write_count")
            return

        if most_similar_record is not None and most_similar_record.utility < 0.5:
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            if new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin:
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_low_quality_replace_count")
                return

        worst_same_signature_record, worst_stats = self._worst_same_signature_record(
            new_record.signature, probe_embedding_list
        )
        if (
            worst_same_signature_record is not None
            and (
                new_stats["mean"]
                > worst_stats["mean"] + self.rfm_same_signature_replace_margin
                or (
                    worst_same_signature_record.utility < self.min_utility
                    and new_stats["positive_ratio"]
                    >= self.rfm_positive_ratio_threshold
                )
            )
        ):
            self._replace_memory(worst_same_signature_record, new_record)
            self._increment_management_metric("rfm_same_signature_replace_count")
            return

        self._increment_management_metric("rfm_controller_skip_count")
        self._increment_management_metric("rfm_write_skip_count")

    def _memory_eviction_score(
        self,
        record: TokenWorthMemoryRecord,
        probe_embedding_list: Sequence[Sequence[float]],
    ) -> float:
        rfm_score = 0.0
        if self._rfm_management_ready(count_metric=False) and len(probe_embedding_list) > 0:
            rfm_score = self._score_stats(
                self._score_list_on_probe_embeddings(record, probe_embedding_list)
            )["mean"]
        utility_score = max(-2.0, min(5.0, record.utility)) / 5.0
        reliability_score = record.reliability()
        access_score = min(1.0, record.access_count / max(1, self.metrics.step))
        return (
            self.eviction_rfm_weight * rfm_score
            + self.eviction_utility_weight * utility_score
            + self.eviction_reliability_weight * reliability_score
            + self.eviction_access_weight * access_score
        )

    def _evict_if_needed(self) -> None:
        if len(self.memory_record_list) <= self.max_memory_count:
            return
        probe_embedding_list = self._eviction_probe_embedding_list()
        evict_count = len(self.memory_record_list) - self.max_memory_count
        candidate_record_list = list(self.memory_record_list)
        candidate_record_list.sort(
            key=lambda record: (
                self._memory_is_protected(record),
                self._memory_eviction_score(record, probe_embedding_list),
                record.utility,
                record.help_count - record.harm_count,
                record.access_count,
                record.last_access_step,
            )
        )
        evict_id_set = {record.memory_id for record in candidate_record_list[:evict_count]}
        self.memory_record_list = [
            record for record in self.memory_record_list if record.memory_id not in evict_id_set
        ]
        self.metrics.evict_count += evict_count
        self._increment_management_metric("rfm_evict_count", evict_count)


class TokenWorthUnifiedBankMemoryCallback(
    TokenWorthCoverageBankMemoryCallback
):
    """TokenWorth unified bank controller without a dedicated conflict branch.

    Similar memories are handled by the generic same-signature RFM replacement
    rule instead of a separate high-similarity conflict rule.
    """

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        most_similar_record, similarity = self._find_most_similar_memory(new_record)

        if len(self.memory_record_list) < self.force_write_until_memory_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("warmup_write_count")
            return

        same_signature_count = sum(
            1
            for record in self.memory_record_list
            if record.signature == new_record.signature
        )
        if same_signature_count < self.signature_warmup_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("signature_warmup_write_count")
            return

        if not self._rfm_management_ready():
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_coverage_repair_write_count")
            return

        probe_embedding_list = self._build_probe_embedding_list(new_record)
        new_score_list = self._score_list_on_probe_embeddings(
            new_record, probe_embedding_list
        )
        new_stats = self._score_stats(new_score_list)

        if (
            len(selected_record_list) > 0
            and most_similar_record is not None
            and most_similar_record.memory_id
            in {record.memory_id for record in selected_record_list}
            and similarity >= self.existing_solution_similarity_threshold
            and most_similar_record.utility >= 0.5
        ):
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            win_ratio = self._win_ratio(
                new_score_list, old_score_list, self.rfm_replace_margin
            )
            if (
                new_stats["mean"] > old_stats["mean"] + self.rfm_replace_margin
                and win_ratio >= self.rfm_win_ratio_threshold
            ):
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_existing_replace_count")
            else:
                self.metrics.solved_by_existing_memory_count += 1
                self._increment_management_metric("rfm_existing_skip_count")
            return

        if most_similar_record is None or similarity < self.novelty_threshold:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        best_old_stats = self._best_existing_score_stats(
            probe_embedding_list, new_record.signature
        )
        gain = new_stats["mean"] - best_old_stats["mean"]
        if (
            gain > self.rfm_write_gain_margin
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        active_same_signature_count = self._same_signature_active_count(
            new_record.signature
        )
        if (
            active_same_signature_count < self.min_active_same_signature_count
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_coverage_repair_write_count")
            return

        if most_similar_record is not None and most_similar_record.utility < 0.5:
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            if new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin:
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_low_quality_replace_count")
                return

        worst_same_signature_record, worst_stats = self._worst_same_signature_record(
            new_record.signature, probe_embedding_list
        )
        if (
            worst_same_signature_record is not None
            and (
                new_stats["mean"]
                > worst_stats["mean"] + self.rfm_same_signature_replace_margin
                or (
                    worst_same_signature_record.utility < self.min_utility
                    and new_stats["positive_ratio"]
                    >= self.rfm_positive_ratio_threshold
                )
            )
        ):
            self._replace_memory(worst_same_signature_record, new_record)
            self._increment_management_metric("rfm_same_signature_replace_count")
            return

        self._increment_management_metric("rfm_controller_skip_count")
        self._increment_management_metric("rfm_write_skip_count")


class TokenWorthSignatureFreeMemoryCallback(
    TokenWorthUnifiedBankMemoryCallback
):
    """TokenWorth memory controller with signature usage removed.

    The memory record keeps a constant placeholder signature only for schema
    compatibility. Retrieval, probes, write gain, replacement, and coverage
    repair do not use same-signature groups.
    """

    def _build_signature(self, text: str) -> str:
        return "none"

    def _build_probe_embedding_list(
        self,
        new_record: TokenWorthMemoryRecord,
    ) -> list[list[float]]:
        probe_embedding_list = [new_record.query_embedding]
        probe_embedding_list.extend(
            self.recent_query_embedding_list[-self.recent_probe_count :]
        )
        return probe_embedding_list

    def _best_existing_score_stats(
        self,
        probe_embedding_list: Sequence[Sequence[float]],
        signature: str,
    ) -> dict[str, float]:
        best_stats = {"mean": float("-inf"), "positive_ratio": 0.0, "std": 0.0}
        for record in self.memory_record_list:
            if record.utility < self.min_utility:
                continue
            stats = self._score_stats(
                self._score_list_on_probe_embeddings(record, probe_embedding_list)
            )
            if stats["mean"] > best_stats["mean"]:
                best_stats = stats
        return best_stats

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        most_similar_record, similarity = self._find_most_similar_memory(new_record)

        if len(self.memory_record_list) < self.force_write_until_memory_count:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("warmup_write_count")
            return

        if not self._rfm_management_ready():
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_coverage_repair_write_count")
            return

        probe_embedding_list = self._build_probe_embedding_list(new_record)
        new_score_list = self._score_list_on_probe_embeddings(
            new_record, probe_embedding_list
        )
        new_stats = self._score_stats(new_score_list)

        if most_similar_record is None or similarity < self.novelty_threshold:
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        best_old_stats = self._best_existing_score_stats(
            probe_embedding_list, new_record.signature
        )
        gain = new_stats["mean"] - best_old_stats["mean"]
        if (
            gain > self.rfm_write_gain_margin
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_write_accept_count")
            return

        if most_similar_record is not None and most_similar_record.utility < 0.5:
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            if new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin:
                self._replace_memory(most_similar_record, new_record)
                self._increment_management_metric("rfm_low_quality_replace_count")
                return

        self._increment_management_metric("rfm_controller_skip_count")
        self._increment_management_metric("rfm_write_skip_count")


class _TokenWorthHashingMemoryBase(
    TokenWorthSignatureFreeMemoryCallback
):
    """Internal TokenWorth base with lightweight hashing embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        character_ngram_min: int = 3,
        character_ngram_max: int = 5,
        **kwargs,
    ):
        kwargs.setdefault("embedding_model_path", "")
        super().__init__(**kwargs)
        assert embedding_dim > 0
        assert character_ngram_min > 0
        assert character_ngram_max >= character_ngram_min
        self.embedding_dim = embedding_dim
        self.character_ngram_min = character_ngram_min
        self.character_ngram_max = character_ngram_max

    def _get_embedding_cache_path(self) -> str:
        return os.path.join(
            self.get_state_dir(), "q2m_no_signature_lightweight_embedding_cache.json"
        )

    def _encode_dense_vector(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.embedding_dim)]
        for token in tokenize(text):
            self._add_feature(vector, f"tok:{token}", 1.0)
        normalized_text = " ".join(tokenize(text))
        for ngram_size in range(
            self.character_ngram_min, self.character_ngram_max + 1
        ):
            if len(normalized_text) < ngram_size:
                continue
            weight = 0.35 / math.sqrt(ngram_size)
            for index in range(0, len(normalized_text) - ngram_size + 1):
                self._add_feature(
                    vector,
                    f"chr{ngram_size}:{normalized_text[index:index + ngram_size]}",
                    weight,
                )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return vector
        return [value / norm for value in vector]

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.embedding_dim
        sign = 1.0 if int.from_bytes(digest[4:], "big") % 2 == 0 else -1.0
        vector[bucket] += sign * weight


class TokenWorthBankActionMemoryCallback(
    _TokenWorthHashingMemoryBase
):
    """Bank-level action selector for add, replace, add-then-evict, or skip."""

    def __init__(self, *args, weak_replace_priority_bonus: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_replace_priority_bonus = weak_replace_priority_bonus
        self.bank_management_metrics.setdefault("bank_action_warmup_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_not_ready_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_novelty_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_rfm_gain_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_add_then_evict_count", 0)
        self.bank_management_metrics.setdefault("bank_action_replace_weak_count", 0)
        self.bank_management_metrics.setdefault("bank_action_skip_count", 0)

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        most_similar_record, similarity = self._find_most_similar_memory(new_record)

        if len(self.memory_record_list) < self.force_write_until_memory_count:
            self._execute_bank_action("add", new_record, None)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("warmup_write_count")
            self._increment_management_metric("bank_action_warmup_add_count")
            return

        if not self._rfm_management_ready():
            self._execute_bank_action("add", new_record, None)
            self.metrics.novelty_write_count += 1
            self._increment_management_metric("rfm_coverage_repair_write_count")
            self._increment_management_metric("bank_action_not_ready_add_count")
            return

        probe_embedding_list = self._build_probe_embedding_list(new_record)
        new_score_list = self._score_list_on_probe_embeddings(
            new_record, probe_embedding_list
        )
        new_stats = self._score_stats(new_score_list)
        candidate_list = self._build_bank_action_candidates(
            new_record,
            most_similar_record,
            similarity,
            probe_embedding_list,
            new_score_list,
            new_stats,
        )
        action = max(candidate_list, key=lambda item: (item["priority"], item["score"]))
        self._execute_selected_bank_action(action, new_record)

    def _build_bank_action_candidates(
        self,
        new_record: TokenWorthMemoryRecord,
        most_similar_record: TokenWorthMemoryRecord | None,
        similarity: float,
        probe_embedding_list: Sequence[Sequence[float]],
        new_score_list: Sequence[float],
        new_stats: dict[str, float],
    ) -> list[dict]:
        candidate_list = [{"type": "skip", "priority": 0.0, "score": 0.0}]

        if most_similar_record is None or similarity < self.novelty_threshold:
            candidate_list.append(
                {
                    "type": "add",
                    "reason": "novelty",
                    "priority": 3.0,
                    "score": new_stats["mean"],
                }
            )

        best_old_stats = self._best_existing_score_stats(
            probe_embedding_list, new_record.signature
        )
        gain = new_stats["mean"] - best_old_stats["mean"]
        if (
            gain > self.rfm_write_gain_margin
            and new_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            candidate_list.append(
                {
                    "type": "add",
                    "reason": "rfm_gain",
                    "priority": 2.0,
                    "score": gain,
                }
            )

        if most_similar_record is not None and most_similar_record.utility < 0.5:
            old_score_list = self._score_list_on_probe_embeddings(
                most_similar_record, probe_embedding_list
            )
            old_stats = self._score_stats(old_score_list)
            if new_stats["mean"] >= old_stats["mean"] - self.rfm_skip_margin:
                candidate_list.append(
                    {
                        "type": "replace",
                        "reason": "weak_similar",
                        "target": most_similar_record,
                        "priority": 2.0 + self.weak_replace_priority_bonus,
                        "score": new_stats["mean"] - old_stats["mean"],
                    }
                )

        return candidate_list

    def _execute_selected_bank_action(
        self,
        action: dict,
        new_record: TokenWorthMemoryRecord,
    ) -> None:
        action_type = action["type"]
        if action_type == "add":
            self._execute_bank_action("add", new_record, None)
            self.metrics.novelty_write_count += 1
            if action.get("reason") == "novelty":
                self._increment_management_metric("rfm_write_accept_count")
                self._increment_management_metric("bank_action_novelty_add_count")
            else:
                self._increment_management_metric("rfm_write_accept_count")
                self._increment_management_metric("bank_action_rfm_gain_add_count")
            if len(self.memory_record_list) > self.max_memory_count:
                self._increment_management_metric("bank_action_add_then_evict_count")
            return

        if action_type == "replace":
            target = action.get("target")
            if target is not None:
                self._execute_bank_action("replace", new_record, target)
                self._increment_management_metric("rfm_low_quality_replace_count")
                self._increment_management_metric("bank_action_replace_weak_count")
                return

        self._increment_management_metric("rfm_controller_skip_count")
        self._increment_management_metric("rfm_write_skip_count")
        self._increment_management_metric("bank_action_skip_count")

    def _execute_bank_action(
        self,
        action_type: str,
        new_record: TokenWorthMemoryRecord,
        target_record: TokenWorthMemoryRecord | None,
    ) -> None:
        if action_type == "add":
            self._append_memory(new_record)
            return
        if action_type == "replace" and target_record is not None:
            self._replace_memory(target_record, new_record)


class TokenWorthDenseBankActionMemoryCallback(
    TokenWorthSignatureFreeMemoryCallback
):
    """Bank-level action selector with model embeddings."""

    def __init__(self, *args, weak_replace_priority_bonus: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_replace_priority_bonus = weak_replace_priority_bonus
        self.bank_management_metrics.setdefault("bank_action_warmup_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_not_ready_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_novelty_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_rfm_gain_add_count", 0)
        self.bank_management_metrics.setdefault("bank_action_add_then_evict_count", 0)
        self.bank_management_metrics.setdefault("bank_action_replace_weak_count", 0)
        self.bank_management_metrics.setdefault("bank_action_skip_count", 0)

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        TokenWorthBankActionMemoryCallback._write_update_or_skip_memory(
            self, new_record, selected_record_list
        )

    def _build_bank_action_candidates(
        self,
        new_record: TokenWorthMemoryRecord,
        most_similar_record: TokenWorthMemoryRecord | None,
        similarity: float,
        probe_embedding_list: Sequence[Sequence[float]],
        new_score_list: Sequence[float],
        new_stats: dict[str, float],
    ) -> list[dict]:
        return TokenWorthBankActionMemoryCallback._build_bank_action_candidates(
            self,
            new_record,
            most_similar_record,
            similarity,
            probe_embedding_list,
            new_score_list,
            new_stats,
        )

    def _execute_selected_bank_action(
        self,
        action: dict,
        new_record: TokenWorthMemoryRecord,
    ) -> None:
        TokenWorthBankActionMemoryCallback._execute_selected_bank_action(
            self, action, new_record
        )

    def _execute_bank_action(
        self,
        action_type: str,
        new_record: TokenWorthMemoryRecord,
        target_record: TokenWorthMemoryRecord | None,
    ) -> None:
        TokenWorthBankActionMemoryCallback._execute_bank_action(
            self, action_type, new_record, target_record
        )


class TokenWorthEvaluationAlignedMemoryCallback(
    TokenWorthDenseBankActionMemoryCallback
):
    """OS-format variant that treats evaluator-correct sessions as successful.

    OSInteraction can produce a correct evaluation outcome even when the session
    ends by the round limit instead of an explicit finish action. This adapter
    keeps the bank-action memory policy unchanged and only aligns memory
    feedback/write decisions with the OS task's evaluation format.
    """

    def on_task_complete(self, callback_args) -> None:
        current_session = callback_args.current_session
        is_correct = (
            current_session.evaluation_record.outcome
            == SessionEvaluationOutcome.CORRECT
        )
        selected_record_list = [
            record
            for record in self.memory_record_list
            if record.memory_id in set(self._last_selected_id_list)
        ]
        for record in selected_record_list:
            self._update_memory_feedback(record, is_correct)
            pair_feature = self._last_selected_pair_feature_dict.get(record.memory_id)
            if pair_feature is None:
                pair_feature = self._build_pair_feature(
                    self._current_query_embedding, record
                )
            self._append_pair_record(
                pair_feature,
                1.0 if is_correct else -1.0,
                record.memory_id,
            )
        if (
            len(selected_record_list) > 0
            and len(self.pair_record_list) >= self.min_pair_count_for_rfm
        ):
            if self.metrics.pair_write_count % self.rfm_update_interval == 0:
                self._update_rfm()
        if is_correct:
            new_record = self._build_memory_record_from_session(current_session)
            if new_record is not None:
                self._write_update_or_skip_memory(new_record, selected_record_list)
        self._evict_if_needed()


class TokenWorthOSPolicyMemoryCallback(
    TokenWorthEvaluationAlignedMemoryCallback
):
    """OS-format adapter that stores compact execution policies.

    The memory content avoids prior task objects and stores reusable execution
    discipline inferred from successful bash actions.
    """

    _BASE_POLICY_LINE_LIST = [
        "Use one non-interactive bash action when possible.",
        "Make commands idempotent: create parents first, overwrite safely, then set ownership/permissions.",
        "Verify the requested state explicitly before ending the action.",
    ]

    def _build_memory_record_from_session(
        self, session: Session
    ) -> TokenWorthMemoryRecord | None:
        query = extract_question_from_session(session).strip()
        if len(query) == 0:
            return None
        action_text = self._build_successful_action_text(session)
        if len(action_text) == 0:
            return None
        policy_line_list = self._build_policy_line_list(action_text)
        idiom_line_list = self._build_command_idiom_line_list(action_text)
        snippet_line_list = ["Execution policy:"]
        snippet_line_list.extend(f"- {line}" for line in policy_line_list)
        if len(idiom_line_list) > 0:
            snippet_line_list.append("Command idioms:")
            snippet_line_list.extend(f"- {line}" for line in idiom_line_list)
        snippet = truncate_text("\n".join(snippet_line_list), self.max_snippet_length)
        embedding = self._encode_text(query)
        return TokenWorthMemoryRecord(
            memory_id=self._next_memory_id,
            query=query,
            snippet=snippet,
            search_text=f"{query}\n{self._build_policy_search_text(policy_line_list, idiom_line_list)}\n{self._build_signature(query)}",
            query_embedding=embedding,
            signature=self._build_signature(query),
            created_step=self.metrics.step,
            last_access_step=self.metrics.step,
        )

    def _build_successful_action_text(self, session: Session) -> str:
        action_list: list[str] = []
        for item_index in range(3, session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(item_index)
            if item.role != Role.AGENT:
                continue
            action_text = self._extract_action_text(item.content)
            bash_code = self._extract_bash_code(action_text)
            if len(bash_code) > 0:
                action_list.append(bash_code)
        return "\n".join(action_list[-2:])

    def _extract_bash_code(self, action_text: str) -> str:
        if not action_text.lower().startswith("act: bash"):
            return ""
        match = re.search(r"```(?:bash)?\s*([\s\S]*?)```", action_text, re.I)
        if match is None:
            return ""
        return match.group(1).strip()

    def _build_policy_line_list(self, action_text: str) -> list[str]:
        line_list = list(self._BASE_POLICY_LINE_LIST)
        lowered = action_text.lower()
        if re.search(r"\b(chown|chmod|chgrp)\b", lowered):
            line_list.append("For ownership/permission tasks, apply the change and verify with stat/ls/getent/id.")
        if re.search(r"\b(useradd|usermod|groupadd|chage|chsh)\b", lowered):
            line_list.append("For user/group tasks, prefer idempotent checks and non-interactive account commands.")
        if re.search(r"\b(find|xargs|while read|for .+ in)\b", lowered):
            line_list.append("For batch file tasks, use find/loops with explicit predicates and verify affected paths.")
        if re.search(r"\b(sed|awk|grep|tee|cat >|printf)\b", lowered):
            line_list.append("For text edits, write deterministically and verify exact content or matching lines.")
        if re.search(r"\b(systemctl|service|nohup|sleep|&)\b", lowered):
            line_list.append("For process/service tasks, start non-interactively and verify process/log/state.")
        deduped_line_list: list[str] = []
        for line in line_list:
            if line not in deduped_line_list:
                deduped_line_list.append(line)
        return deduped_line_list[:6]

    def _build_command_idiom_line_list(self, action_text: str) -> list[str]:
        idiom_list: list[str] = []
        for name, pattern in [
            ("mkdir -p before writing nested files", r"\bmkdir\s+-p\b"),
            ("set -e or && to stop on failures", r"\bset\s+-e\b|&&"),
            ("use install/chmod/chown for permissions", r"\binstall\b|\bchmod\b|\bchown\b|\bchgrp\b"),
            ("use grep/stat/ls/find/id/getent for verification", r"\bgrep\b|\bstat\b|\bls\b|\bfind\b|\bid\b|\bgetent\b"),
            ("avoid interactive editors; create files with printf/cat/tee", r"\bprintf\b|\btee\b|cat\s*>"),
        ]:
            if re.search(pattern, action_text):
                idiom_list.append(name)
        return idiom_list[:4]

    def _build_policy_search_text(
        self,
        policy_line_list: Sequence[str],
        idiom_line_list: Sequence[str],
    ) -> str:
        return " ".join(list(policy_line_list) + list(idiom_line_list))


class TokenWorthOSIdiomBuilderMemoryCallback(
    TokenWorthOSPolicyMemoryCallback
):
    """OS-format adapter that stores only category-specific execution idioms."""

    def _build_memory_record_from_session(
        self, session: Session
    ) -> TokenWorthMemoryRecord | None:
        query = extract_question_from_session(session).strip()
        if len(query) == 0:
            return None
        action_text = self._build_successful_action_text(session)
        if len(action_text) == 0:
            return None
        idiom_line_list = self._build_category_idiom_line_list(action_text)
        if len(idiom_line_list) == 0:
            idiom_line_list = self._build_command_idiom_line_list(action_text)
        if len(idiom_line_list) == 0:
            return None
        snippet_line_list = ["Retrieved execution idiom:"]
        snippet_line_list.extend(f"- {line}" for line in idiom_line_list[:4])
        snippet = truncate_text("\n".join(snippet_line_list), self.max_snippet_length)
        embedding = self._encode_text(query)
        return TokenWorthMemoryRecord(
            memory_id=self._next_memory_id,
            query=query,
            snippet=snippet,
            search_text=f"{query}\n{' '.join(idiom_line_list)}\n{self._build_signature(query)}",
            query_embedding=embedding,
            signature=self._build_signature(query),
            created_step=self.metrics.step,
            last_access_step=self.metrics.step,
        )

    def _build_category_idiom_line_list(self, action_text: str) -> list[str]:
        lowered = action_text.lower()
        idiom_list: list[str] = []
        if re.search(r"\b(chown|chmod|chgrp|stat)\b", lowered):
            idiom_list.append(
                "Permissions/ownership: apply chmod/chown/chgrp after creating paths; verify with stat -c or ls -l."
            )
        if re.search(r"\b(useradd|usermod|groupadd|chage|chsh|getent|id)\b", lowered):
            idiom_list.append(
                "Users/groups: use getent/id guards with useradd/groupadd/usermod; avoid chsh interaction by using usermod -s."
            )
        if re.search(r"\b(find|xargs|while read|for .+ in)\b", lowered):
            idiom_list.append(
                "Batch files: use find with precise predicates and a loop; create destination dirs before moving/copying."
            )
        if re.search(r"\b(sed|awk|grep|tee|printf|cat >)\b", lowered):
            idiom_list.append(
                "Text edits: use sed/awk/printf/tee deterministically; verify final content with grep/cat."
            )
        if re.search(r"\b(ln\s+-s|readlink)\b", lowered):
            idiom_list.append(
                "Symlinks: create parent dir, replace stale links with ln -sfn or rm first, then verify with readlink."
            )
        if re.search(r"\b(nohup|systemctl|service|sleep|pgrep|&)\b", lowered):
            idiom_list.append(
                "Processes/services: start non-interactively in background when needed; verify with pgrep/log/state checks."
            )
        return idiom_list


class TokenWorthMemoryCallback(_TokenWorthHashingMemoryBase):
    """Paper §4 and Appendix B: learned base plus two-timescale residual RFM."""

    def __init__(
        self,
        *args,
        krr_update_interval: int = 75,
        agop_update_interval: int = 225,
        krr_buffer_size: int = 128,
        agop_buffer_size: int = 256,
        rfm_influence: float = 0.10,
        rfm_influence_warmup_updates: int = 3,
        rfm_residual_clip: float = 1.0,
        base_buffer_size: int = 256,
        base_update_interval: int = 50,
        base_regularization: float = 0.1,
        base_smoothing: float = 0.5,
        replacement_value_margin: float = 0.10,
        **kwargs,
    ):
        assert krr_update_interval > 0
        assert agop_update_interval > 0
        assert krr_buffer_size > 0
        assert agop_buffer_size >= krr_buffer_size
        assert rfm_influence >= 0.0
        assert rfm_influence_warmup_updates > 0
        assert rfm_residual_clip > 0.0
        assert base_buffer_size > 0 and base_update_interval > 0
        assert base_regularization > 0.0 and 0.0 <= base_smoothing <= 1.0
        assert replacement_value_margin >= 0.0

        kwargs.setdefault("use_evaluator_outcome_only", True)
        kwargs.setdefault("need_memory_gate_enabled", False)
        kwargs.setdefault("rfm_positive_ratio_threshold", 0.45)
        kwargs.setdefault("rfm_write_gain_margin", 0.02)
        # The parent trigger is disabled; feedback appends drive the two clocks.
        kwargs["pair_buffer_size"] = krr_buffer_size
        kwargs["rfm_update_interval"] = 1_000_000_000
        super().__init__(*args, **kwargs)

        self.krr_update_interval = krr_update_interval
        self.agop_update_interval = agop_update_interval
        self.krr_buffer_size = krr_buffer_size
        self.agop_buffer_size = agop_buffer_size
        self.rfm_influence = rfm_influence
        self.rfm_influence_warmup_updates = rfm_influence_warmup_updates
        self.rfm_residual_clip = rfm_residual_clip

        self.agop_pair_record_list: list[TokenWorthPairRecord] = []
        self.krr_train_record_list: list[TokenWorthPairRecord] = []
        self.hybrid_krr_refit_count = 0
        self.hybrid_agop_update_count = 0
        self.base_buffer_size = base_buffer_size
        self.base_update_interval = base_update_interval
        self.base_regularization = base_regularization
        self.base_smoothing = base_smoothing
        self.replacement_value_margin = replacement_value_margin
        self.base_prior = [1.0, 0.10, 0.10, 0.02]
        self.base_weight_list = list(self.base_prior)
        self.base_feedback_record_list: list[TokenWorthBaseFeedbackRecord] = []
        self.base_refit_count = 0
        self._last_selected_base_score_dict: dict[int, float] = {}
        self._prediction_cache = None

    def _is_successful_session(self, session: Session) -> bool:
        # Appendix A.4 defines reward by externally verified final correctness.
        # A correct state remains successful when an interaction/output limit
        # ends the execution before the explicit finish action.
        return session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        self._last_selected_base_score_dict = {}
        super().on_task_reset(callback_args)

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        # Service/evaluator failures are not environment rewards and must not
        # become negative training examples for the worth estimator.
        if session.evaluation_record.outcome not in (
            SessionEvaluationOutcome.CORRECT,
            SessionEvaluationOutcome.INCORRECT,
        ) or session.sample_status in (
            SampleStatus.TASK_ENVIRONMENT_ERROR,
            SampleStatus.TASK_UNKNOWN_ERROR,
            SampleStatus.AGENT_UNKNOWN_ERROR,
            SampleStatus.AGENT_OUT_OF_MEMORY,
        ):
            return
        is_correct = self._is_successful_session(session)
        selected = [
            record
            for record in self.memory_record_list
            if record.memory_id in self._last_selected_id_list
        ]
        for record in selected:
            # Cached retrieval-time features and scores precede access and
            # reward updates, avoiding label leakage in online supervision.
            feature = self._last_selected_pair_feature_dict[record.memory_id]
            self._append_pair_record(
                feature, 1.0 if is_correct else -1.0, record.memory_id
            )
            self._update_memory_feedback(record, is_correct)
        if is_correct:
            new_record = self._build_memory_record_from_session(session)
            if new_record is not None:
                self._write_update_or_skip_memory(new_record, selected)
        self._evict_if_needed()
        if selected and self._current_query_embedding:
            self.recent_query_embedding_list.append(list(self._current_query_embedding))
            self.recent_query_embedding_list = self.recent_query_embedding_list[
                -self.recent_probe_count :
            ]

    def _get_paper_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "tokenworth_paper_state.json")

    def _get_hybrid_agop_pair_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_hybrid_agop_pair_records.json")

    def _get_hybrid_krr_train_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_hybrid_krr_train_records.json")

    def _get_hybrid_metrics_path(self) -> str:
        return os.path.join(self.get_state_dir(), "q2m_hybrid_rfm_metrics.json")

    def restore_state(self) -> None:
        # Raw-label checkpoints from the earlier implementation cannot be
        # interpreted as residual-learning checkpoints of the paper algorithm.
        paper_path = self._get_paper_state_path()
        if not os.path.exists(paper_path):
            raise ValueError(
                "TokenWorth checkpoint lacks paper algorithm state; start a new run."
            )
        for required_path in (
            self._get_memory_path(),
            self._get_pair_path(),
            self._get_agop_path(),
            self._get_alpha_path(),
            self._get_metrics_path(),
            self._get_recent_probe_path(),
            self._get_management_metrics_path(),
            self._get_hybrid_agop_pair_path(),
            self._get_hybrid_krr_train_path(),
            self._get_hybrid_metrics_path(),
        ):
            if not os.path.exists(required_path):
                raise ValueError(f"Incomplete TokenWorth checkpoint: {required_path}")
        super().restore_state()
        with open(paper_path, encoding="utf-8") as file:
            paper_state = json.load(file)
        if paper_state.get("schema_version") != 1:
            raise ValueError("Unsupported TokenWorth paper checkpoint version.")
        self.base_weight_list = [float(value) for value in paper_state["base_weights"]]
        self.base_feedback_record_list = [
            TokenWorthBaseFeedbackRecord.model_validate(item)
            for item in paper_state["base_feedback"]
        ]
        self.base_refit_count = int(paper_state["base_refit_count"])
        saved_next_id = int(paper_state["next_memory_id"])
        for path, target in [
            (self._get_hybrid_agop_pair_path(), self.agop_pair_record_list),
            (self._get_hybrid_krr_train_path(), self.krr_train_record_list),
        ]:
            if not os.path.exists(path):
                raise ValueError(f"Incomplete TokenWorth checkpoint: {path}")
            target.clear()
            target.extend(
                TokenWorthPairRecord.model_validate(item)
                for item in json.load(open(path, encoding="utf-8"))
            )
        if os.path.exists(self._get_hybrid_metrics_path()):
            metrics = json.load(open(self._get_hybrid_metrics_path(), encoding="utf-8"))
            self.hybrid_krr_refit_count = int(metrics.get("krr_refit_count", 0))
            self.hybrid_agop_update_count = int(metrics.get("agop_update_count", 0))
        if not os.path.exists(self._get_hybrid_metrics_path()):
            raise ValueError(
                "Incomplete TokenWorth checkpoint: hybrid metrics missing."
            )
        for field, count in (
            ("krr_train_count", len(self.krr_train_record_list)),
            ("krr_feedback_count", len(self.pair_record_list)),
            ("agop_feedback_count", len(self.agop_pair_record_list)),
        ):
            if metrics[field] != count:
                raise ValueError(f"Corrupt TokenWorth checkpoint: {field} mismatch.")
        if len(self.kernel_alpha_list) != len(self.krr_train_record_list):
            raise ValueError(
                "Corrupt TokenWorth checkpoint: KRR coefficients/training rows mismatch."
            )
        if (self.hybrid_krr_refit_count > 0) != bool(self.krr_train_record_list):
            raise ValueError(
                "Corrupt TokenWorth checkpoint: KRR update count/training rows mismatch."
            )
        all_records = (
            self.memory_record_list
            + self.pair_record_list
            + self.agop_pair_record_list
            + self.krr_train_record_list
        )
        if saved_next_id < max(
            (record.memory_id + 1 for record in all_records), default=0
        ):
            raise ValueError(
                "Corrupt TokenWorth checkpoint: next memory ID would be reused."
            )
        self._next_memory_id = saved_next_id
        self._validate_paper_state()

    def _validate_paper_state(self) -> None:
        if (
            self.base_refit_count
            != self.metrics.pair_write_count // self.base_update_interval
        ):
            raise ValueError(
                "Corrupt TokenWorth checkpoint: base update counter mismatch."
            )
        if len(self.base_feedback_record_list) != min(
            self.metrics.pair_write_count, self.base_buffer_size
        ):
            raise ValueError(
                "Corrupt TokenWorth checkpoint: base feedback count mismatch."
            )
        if len(self.base_weight_list) != 4 or any(
            not math.isfinite(value) or value < 0 or value > upper
            for value, upper in zip(self.base_weight_list, [2.0, 1.0, 1.0, 1.0])
        ):
            raise ValueError("Corrupt TokenWorth checkpoint: invalid base weights.")
        if len(self.base_feedback_record_list) > self.base_buffer_size:
            raise ValueError(
                "Corrupt TokenWorth checkpoint: base buffer exceeds capacity."
            )
        for record in self.base_feedback_record_list:
            if (
                len(record.feature) != 4
                or record.label not in (-1.0, 1.0)
                or not all(map(math.isfinite, record.feature))
            ):
                raise ValueError(
                    "Corrupt TokenWorth checkpoint: invalid base feedback."
                )
        if len(self.memory_record_list) > self.max_memory_count or len(
            {r.memory_id for r in self.memory_record_list}
        ) != len(self.memory_record_list):
            raise ValueError("Corrupt TokenWorth checkpoint: invalid memory bank.")
        if (
            len(self.pair_record_list) > self.krr_buffer_size
            or len(self.agop_pair_record_list) > self.agop_buffer_size
            or len(self.krr_train_record_list) > self.krr_buffer_size
        ):
            raise ValueError(
                "Corrupt TokenWorth checkpoint: residual buffer exceeds capacity."
            )
        for record in (
            self.pair_record_list
            + self.agop_pair_record_list
            + self.krr_train_record_list
        ):
            if (
                len(record.pair_feature) != self.embedding_dim + 5
                or not all(map(math.isfinite, record.pair_feature))
                or not math.isfinite(record.label)
            ):
                raise ValueError(
                    "Corrupt TokenWorth checkpoint: invalid residual feedback."
                )
        if self.agop_diag_list and (
            len(self.agop_diag_list) != self.embedding_dim + 5
            or any(
                not math.isfinite(value) or value <= 0 for value in self.agop_diag_list
            )
        ):
            raise ValueError("Corrupt TokenWorth checkpoint: invalid feature metric.")
        if any(not math.isfinite(value) for value in self.kernel_alpha_list):
            raise ValueError("Corrupt TokenWorth checkpoint: invalid KRR coefficient.")

    def on_state_save(self, callback_args) -> None:
        super().on_state_save(callback_args)
        with open(self._get_paper_state_path(), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "schema_version": 1,
                    "next_memory_id": self._next_memory_id,
                    "base_weights": self.base_weight_list,
                    "base_feedback": [
                        record.model_dump() for record in self.base_feedback_record_list
                    ],
                    "base_refit_count": self.base_refit_count,
                },
                file,
                indent=2,
            )
        json.dump(
            [record.model_dump() for record in self.agop_pair_record_list],
            open(self._get_hybrid_agop_pair_path(), "w", encoding="utf-8"),
            indent=2,
        )
        json.dump(
            [record.model_dump() for record in self.krr_train_record_list],
            open(self._get_hybrid_krr_train_path(), "w", encoding="utf-8"),
            indent=2,
        )
        json.dump(
            {
                "krr_refit_count": self.hybrid_krr_refit_count,
                "agop_update_count": self.hybrid_agop_update_count,
                "effective_rfm_influence": self._effective_rfm_influence(),
                "krr_train_count": len(self.krr_train_record_list),
                "krr_feedback_count": len(self.pair_record_list),
                "agop_feedback_count": len(self.agop_pair_record_list),
            },
            open(self._get_hybrid_metrics_path(), "w", encoding="utf-8"),
            indent=2,
        )

    def _append_pair_record(
        self,
        pair_feature: list[float],
        label: float,
        memory_id: int,
    ) -> None:
        base_feature = self._base_features_from_pair(pair_feature)
        base_score = self._last_selected_base_score_dict.get(
            memory_id, self._base_pair_score(pair_feature)
        )
        residual_target = label - math.tanh(base_score / 2.0)
        self.base_feedback_record_list.append(
            TokenWorthBaseFeedbackRecord(feature=base_feature, label=label)
        )
        self.base_feedback_record_list = self.base_feedback_record_list[
            -self.base_buffer_size :
        ]
        record = TokenWorthPairRecord(
            pair_feature=pair_feature,
            label=residual_target,
            memory_id=memory_id,
            created_step=self.metrics.step,
        )
        self.pair_record_list.append(record)
        self.pair_record_list = self.pair_record_list[-self.krr_buffer_size :]
        self.agop_pair_record_list.append(record.model_copy(deep=True))
        self.agop_pair_record_list = self.agop_pair_record_list[
            -self.agop_buffer_size :
        ]
        if label > 0:
            self.metrics.positive_pair_count += 1
        else:
            self.metrics.negative_pair_count += 1
        self.metrics.pair_write_count += 1

        pair_count = self.metrics.pair_write_count
        if pair_count % self.base_update_interval == 0:
            self._refit_base_scorer()
        agop_due = pair_count % self.agop_update_interval == 0
        krr_due = pair_count % self.krr_update_interval == 0
        if agop_due and len(self.agop_pair_record_list) >= self.min_pair_count_for_rfm:
            self._update_agop_metric()
        if (agop_due or krr_due) and len(
            self.pair_record_list
        ) >= self.min_pair_count_for_rfm:
            self._refit_main_krr()

    def _refit_base_scorer(self) -> None:
        """Deterministic Newton solve of Appendix B's convex logistic objective."""
        import numpy as np

        x = np.asarray(
            [record.feature for record in self.base_feedback_record_list], dtype=float
        )
        y = np.asarray(
            [record.label for record in self.base_feedback_record_list], dtype=float
        )
        prior = np.asarray(self.base_prior, dtype=float)
        weights = np.asarray(self.base_weight_list, dtype=float)

        def objective(candidate):
            return float(
                np.logaddexp(0.0, -y * (x @ candidate)).sum()
                + self.base_regularization * np.sum((candidate - prior) ** 2)
            )

        for _ in range(100):
            margin = y * (x @ weights)
            wrong_probability = np.exp(-np.logaddexp(0.0, margin))
            gradient = x.T @ (
                -y * wrong_probability
            ) + 2.0 * self.base_regularization * (weights - prior)
            if np.max(np.abs(gradient)) < 1e-9:
                break
            curvature = wrong_probability * (1.0 - wrong_probability)
            hessian = x.T @ (
                curvature[:, None] * x
            ) + 2.0 * self.base_regularization * np.eye(4)
            direction = np.linalg.solve(hessian, gradient)
            step = 1.0
            old_loss = objective(weights)
            for _ in range(40):
                candidate = weights - step * direction
                if objective(candidate) <= old_loss - 1e-4 * step * float(
                    gradient @ direction
                ) + 1e-14 * max(1.0, abs(old_loss)):
                    weights = candidate
                    break
                step *= 0.5
            else:
                raise RuntimeError(
                    "TokenWorth base optimizer failed to find a descent step."
                )
        else:
            raise RuntimeError("TokenWorth base optimizer did not converge.")
        smoothed = (1.0 - self.base_smoothing) * np.asarray(
            self.base_weight_list
        ) + self.base_smoothing * weights
        self.base_weight_list = (
            np.clip(smoothed, 0.0, [2.0, 1.0, 1.0, 1.0]).astype(float).tolist()
        )
        self.base_refit_count += 1

    def _fit_kernel_ridge(self, record_list, metric_diag):
        import numpy as np

        z_matrix = np.asarray(
            [record.pair_feature for record in record_list], dtype=float
        )
        y_vector = np.asarray([record.label for record in record_list], dtype=float)
        diff = z_matrix[:, None, :] - z_matrix[None, :, :]
        distance = np.sqrt(np.maximum(np.sum(metric_diag * diff * diff, axis=2), 0.0))
        kernel_matrix = np.exp(-self.kernel_gamma * distance)
        regularized = kernel_matrix + self.ridge_lambda * np.eye(len(z_matrix))
        try:
            alpha = np.linalg.solve(regularized, y_vector)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(regularized, y_vector, rcond=None)[0]
        return z_matrix, alpha

    def _refit_main_krr(self) -> None:
        import numpy as np

        if len(self.pair_record_list) < self.min_pair_count_for_rfm:
            return
        feature_dim = len(self.pair_record_list[-1].pair_feature)
        self._ensure_agop_diag(feature_dim)
        metric_diag = np.asarray(self.agop_diag_list, dtype=float)
        self.krr_train_record_list = [
            record.model_copy(deep=True) for record in self.pair_record_list
        ]
        _, alpha = self._fit_kernel_ridge(self.krr_train_record_list, metric_diag)
        self.kernel_alpha_list = alpha.astype(float).tolist()
        self.hybrid_krr_refit_count += 1
        self.metrics.rfm_update_count += 1

    def _update_agop_metric(self) -> None:
        import numpy as np

        if len(self.agop_pair_record_list) < self.min_pair_count_for_rfm:
            return
        if not self.kernel_alpha_list:
            self._refit_main_krr()
        metric_diag = np.asarray(self.agop_diag_list, dtype=float)
        train = np.asarray(
            [record.pair_feature for record in self.krr_train_record_list], dtype=float
        )
        points = np.asarray(
            [record.pair_feature for record in self.agop_pair_record_list], dtype=float
        )
        alpha = np.asarray(self.kernel_alpha_list, dtype=float)
        gradients = []
        for point in points:
            diff = point[None, :] - train
            distance = np.sqrt(
                np.maximum(np.sum(metric_diag * diff * diff, axis=1), 0.0)
            )
            # The Laplacian cusp at an identical point uses the zero symmetric
            # subgradient. Other distances use the exact current fast predictor.
            coefficient = np.zeros_like(distance)
            nonzero = distance > 0.0
            coefficient[nonzero] = (
                -self.kernel_gamma
                * alpha[nonzero]
                * np.exp(-self.kernel_gamma * distance[nonzero])
                / distance[nonzero]
            )
            gradients.append(np.sum(coefficient[:, None] * metric_diag * diff, axis=0))
        gradient_matrix = np.asarray(gradients, dtype=float)
        agop_diag = np.mean(gradient_matrix * gradient_matrix, axis=0)
        updated = (
            1.0 - self.agop_smoothing
        ) * metric_diag + self.agop_smoothing * agop_diag
        updated = np.clip(updated, self.agop_min_weight, self.agop_max_weight)
        updated /= float(np.mean(updated))
        self.agop_diag_list = updated.astype(float).tolist()
        self.hybrid_agop_update_count += 1

    def _update_rfm(self) -> None:
        self._update_agop_metric()
        self._refit_main_krr()

    def _base_pair_score(self, pair_feature: Sequence[float]) -> float:
        return float(
            sum(
                weight * feature
                for weight, feature in zip(
                    self.base_weight_list, self._base_features_from_pair(pair_feature)
                )
            )
        )

    @staticmethod
    def _base_features_from_pair(pair_feature: Sequence[float]) -> list[float]:
        if len(pair_feature) < 5:
            raise ValueError(
                "TokenWorth pair representation needs five scalar features."
            )
        return [pair_feature[-5], pair_feature[-4], pair_feature[-3], pair_feature[-2]]

    def _build_pair_feature(self, query_embedding, record) -> list[float]:
        gap = max(0, self.metrics.step - record.last_access_step)
        age = max(1, self.metrics.step - record.created_step + 1)
        feature = [
            abs(q - m) for q, m in zip(query_embedding, record.query_embedding)
        ] + [
            self._cosine(query_embedding, record.query_embedding),
            record.utility,
            record.reliability(),
            math.exp(-0.02 * gap),
            record.access_count / age,
        ]
        self._ensure_agop_diag(len(feature))
        return feature

    def _kernel(self, left_feature, right_feature) -> float:
        if len(left_feature) != len(right_feature) or len(left_feature) != len(
            self.agop_diag_list
        ):
            raise ValueError("TokenWorth kernel feature dimensions do not match.")
        distance_square = sum(
            weight * (left - right) ** 2
            for weight, left, right in zip(
                self.agop_diag_list, left_feature, right_feature
            )
        )
        return math.exp(-self.kernel_gamma * math.sqrt(max(0.0, distance_square)))

    def _ensure_agop_diag(self, dimension: int) -> None:
        if dimension != self.embedding_dim + 5:
            raise ValueError(
                "TokenWorth pair feature schema does not match the encoder."
            )
        if not self.agop_diag_list:
            self.agop_diag_list = [1.0] * dimension
        elif len(self.agop_diag_list) != dimension:
            raise ValueError("TokenWorth metric dimension changed during a run.")

    def _effective_rfm_influence(self) -> float:
        warmup_ratio = min(
            1.0,
            self.hybrid_krr_refit_count / self.rfm_influence_warmup_updates,
        )
        return self.rfm_influence * warmup_ratio

    def _predict_rfm_residual(self, pair_feature: Sequence[float]) -> float:
        import numpy as np

        if len(self.kernel_alpha_list) != len(self.krr_train_record_list):
            raise ValueError("TokenWorth KRR coefficients/training rows mismatch.")
        if not self.kernel_alpha_list:
            return 0.0
        cache = self._prediction_cache
        if (
            cache is None
            or cache[0] is not self.krr_train_record_list
            or cache[1] is not self.kernel_alpha_list
        ):
            cache = (
                self.krr_train_record_list,
                self.kernel_alpha_list,
                np.asarray(
                    [record.pair_feature for record in self.krr_train_record_list],
                    dtype=float,
                ),
                np.asarray(self.kernel_alpha_list, dtype=float),
            )
            self._prediction_cache = cache
        difference = cache[2] - np.asarray(pair_feature, dtype=float)
        distances = np.sqrt(
            np.maximum(
                np.sum(
                    np.asarray(self.agop_diag_list) * difference * difference, axis=1
                ),
                0.0,
            )
        )
        return float(cache[3] @ np.exp(-self.kernel_gamma * distances))

    def _predict_pair_score(self, pair_feature: Sequence[float]) -> float:
        return self._base_pair_score(pair_feature) + (
            self._effective_rfm_influence() * self._predict_rfm_residual(pair_feature)
        )

    def _select_memory_records(self, query: str) -> list[TokenWorthMemoryRecord]:
        candidate_record_list = list(self.memory_record_list)
        candidate_indices = rank_texts_by_relevance(
            query,
            [record.search_text for record in candidate_record_list],
            min(self.candidate_count, len(candidate_record_list)),
        )
        candidate_record_list = [
            candidate_record_list[index] for index in candidate_indices
        ]
        scored_record_list = []
        for record in candidate_record_list:
            pair_feature = self._build_pair_feature(
                self._current_query_embedding, record
            )
            score = self._predict_pair_score(pair_feature)
            scored_record_list.append((score, record, pair_feature))
        scored_record_list.sort(
            key=lambda item: (item[0], -item[1].memory_id), reverse=True
        )

        selected_record_list = []
        for score, record, pair_feature in scored_record_list:
            selected_record_list.append(record)
            self._last_selected_pair_feature_dict[record.memory_id] = pair_feature
            self._last_selected_base_score_dict[record.memory_id] = (
                self._base_pair_score(pair_feature)
            )
            record.access_count += 1
            record.last_access_step = self.metrics.step
            if len(selected_record_list) >= self.top_k:
                break
        return selected_record_list

    def _write_update_or_skip_memory(self, new_record, selected_record_list) -> None:
        if len(self.memory_record_list) < self.max_memory_count:
            self._append_memory(new_record)
            self._increment_management_metric("warmup_write_count")
            return
        nearest, similarity = self._find_most_similar_memory(new_record)
        probes = self._build_probe_embedding_list(new_record)
        candidate_stats = self._score_stats(
            self._score_list_on_probe_embeddings(new_record, probes)
        )
        novelty = 1.0 - similarity
        represented_value = max(
            self._score_stats(self._score_list_on_probe_embeddings(record, probes))[
                "mean"
            ]
            for record in self.memory_record_list
        )
        if novelty > self.novelty_threshold or (
            candidate_stats["mean"] > represented_value + self.rfm_write_gain_margin
            and candidate_stats["positive_ratio"] >= self.rfm_positive_ratio_threshold
        ):
            self._append_memory(new_record)
            self.metrics.novelty_write_count += int(novelty > self.novelty_threshold)
            self._increment_management_metric("rfm_write_accept_count")
            return
        if nearest is not None and nearest.utility < 0.5:
            old_value = self._score_stats(
                self._score_list_on_probe_embeddings(nearest, probes)
            )["mean"]
            if candidate_stats["mean"] > old_value + self.replacement_value_margin:
                self._replace_memory(nearest, new_record)
                self._increment_management_metric("rfm_low_quality_replace_count")
                return
        self._increment_management_metric("rfm_write_skip_count")

    def _memory_eviction_score(self, record, probe_embedding_list) -> float:
        return self._score_stats(
            self._score_list_on_probe_embeddings(record, probe_embedding_list)
        )["mean"]

    def _evict_if_needed(self) -> None:
        if len(self.memory_record_list) <= self.max_memory_count:
            return
        probes = (
            [self._current_query_embedding] if self._current_query_embedding else []
        )
        probes += self.recent_query_embedding_list[-self.recent_probe_count :]
        if not probes:
            raise ValueError(
                "TokenWorth eviction requires a current or previous query."
            )
        excess = len(self.memory_record_list) - self.max_memory_count
        evicted = sorted(
            self.memory_record_list,
            key=lambda record: (
                self._memory_eviction_score(record, probes),
                record.memory_id,
            ),
        )[:excess]
        evicted_ids = {record.memory_id for record in evicted}
        self.memory_record_list = [
            record
            for record in self.memory_record_list
            if record.memory_id not in evicted_ids
        ]
        self.metrics.evict_count += excess
        self._increment_management_metric("rfm_evict_count", excess)


class TokenWorthOSIdiomMemoryCallback(TokenWorthMemoryCallback):
    """Hybrid RFM ranking with task-level OS evaluator feedback and idioms."""

    def __init__(
        self,
        *args,
        weak_replace_priority_bonus: float = 0.01,
        decouple_bank_ready_from_rfm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weak_replace_priority_bonus = weak_replace_priority_bonus
        self.decouple_bank_ready_from_rfm = decouple_bank_ready_from_rfm
        for name in (
            "bank_action_warmup_add_count",
            "bank_action_not_ready_add_count",
            "bank_action_novelty_add_count",
            "bank_action_rfm_gain_add_count",
            "bank_action_add_then_evict_count",
            "bank_action_replace_weak_count",
            "bank_action_skip_count",
            "bank_feedback_ready_count",
        ):
            self.bank_management_metrics.setdefault(name, 0)

    def on_task_complete(self, callback_args) -> None:
        super().on_task_complete(callback_args)

    def _rfm_management_ready(self, *, count_metric: bool = True) -> bool:
        if not self.decouple_bank_ready_from_rfm:
            return super()._rfm_management_ready(count_metric=count_metric)
        ready = (
            self.metrics.pair_write_count >= self.rfm_management_min_pair_count
            and self.metrics.positive_pair_count
            >= self.rfm_management_min_positive_count
            and self.metrics.negative_pair_count
            >= self.rfm_management_min_negative_count
        )
        if ready and count_metric:
            self._increment_management_metric("rfm_ready_count")
            self._increment_management_metric("bank_feedback_ready_count")
        return ready

    def _write_update_or_skip_memory(
        self,
        new_record: TokenWorthMemoryRecord,
        selected_record_list: Sequence[TokenWorthMemoryRecord],
    ) -> None:
        super()._write_update_or_skip_memory(new_record, selected_record_list)

    def _build_bank_action_candidates(
        self,
        new_record: TokenWorthMemoryRecord,
        most_similar_record: TokenWorthMemoryRecord | None,
        similarity: float,
        probe_embedding_list: Sequence[Sequence[float]],
        new_score_list: Sequence[float],
        new_stats: dict[str, float],
    ) -> list[dict]:
        return TokenWorthBankActionMemoryCallback._build_bank_action_candidates(
            self,
            new_record,
            most_similar_record,
            similarity,
            probe_embedding_list,
            new_score_list,
            new_stats,
        )

    def _execute_selected_bank_action(
        self,
        action: dict,
        new_record: TokenWorthMemoryRecord,
    ) -> None:
        TokenWorthBankActionMemoryCallback._execute_selected_bank_action(
            self, action, new_record
        )

    def _execute_bank_action(
        self,
        action_type: str,
        new_record: TokenWorthMemoryRecord,
        target_record: TokenWorthMemoryRecord | None,
    ) -> None:
        TokenWorthBankActionMemoryCallback._execute_bank_action(
            self, action_type, new_record, target_record
        )

    def _select_memory_records(self, query: str) -> list[TokenWorthMemoryRecord]:
        return super()._select_memory_records(query)

    def _build_memory_record_from_session(
        self, session: Session
    ) -> TokenWorthMemoryRecord | None:
        return TokenWorthOSIdiomBuilderMemoryCallback._build_memory_record_from_session(
            self, session
        )

    def _build_successful_action_text(self, session: Session) -> str:
        return TokenWorthOSPolicyMemoryCallback._build_successful_action_text(
            self, session
        )

    def _extract_bash_code(self, action_text: str) -> str:
        return TokenWorthOSPolicyMemoryCallback._extract_bash_code(self, action_text)

    def _build_command_idiom_line_list(self, action_text: str) -> list[str]:
        return TokenWorthOSPolicyMemoryCallback._build_command_idiom_line_list(
            self, action_text
        )

    def _build_category_idiom_line_list(self, action_text: str) -> list[str]:
        return TokenWorthOSIdiomBuilderMemoryCallback._build_category_idiom_line_list(
            self, action_text
        )

    def _is_successful_session(self, session: Session) -> bool:
        return session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
