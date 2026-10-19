import json
import math
import re
from enum import StrEnum
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel

from src.typings import Role, Session, SessionEvaluationOutcome

from .lifelong_memory_utils import (
    ExperienceRecord,
    build_retrieval_text,
    build_session_snippet,
    bm25_rank,
    extract_question_from_session,
    normalize_memory_text,
    tokenize,
    truncate_text,
)


class ProcedureMemoryKind(StrEnum):
    DETAILED = "detailed"
    ABSTRACT = "abstract"
    CAUTION = "caution"


class ProcedureMemoryStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


_RETRIEVAL_STOPWORD_SET = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "has",
    "have",
    "headers",
    "in",
    "is",
    "it",
    "name",
    "of",
    "on",
    "or",
    "return",
    "table",
    "that",
    "the",
    "this",
    "to",
    "where",
    "with",
}


class Mem0Record(BaseModel):
    memory: str
    category: str
    entities: list[str]
    source_query: str
    outcome: SessionEvaluationOutcome
    created_step: int
    last_access_step: int
    mentions: int
    importance: float

    def get_search_text(self) -> str:
        entity_text = " ".join(self.entities)
        return build_retrieval_text(
            f"{self.source_query}\n{self.category}\n{entity_text}\n{self.memory}"
        )


class MemPRecord(BaseModel):
    memory: str
    kind: ProcedureMemoryKind
    status: ProcedureMemoryStatus
    source_query: str
    created_step: int
    last_access_step: int
    support_count: int
    contradiction_count: int

    def get_search_text(self) -> str:
        return f"{self.source_query}\n{self.kind.value}\n{self.memory}"

    def get_render_text(self) -> str:
        return f"[{self.kind.value}] {self.memory}"

    def get_strength(self) -> float:
        return float(self.support_count) - 0.75 * float(self.contradiction_count)


class MemRLEpisodeRecord(BaseModel):
    query: str
    snippet: str
    search_text: str
    outcome: SessionEvaluationOutcome
    q_value: float
    retrieve_count: int
    success_count: int
    failure_count: int
    created_step: int
    last_access_step: int


def extract_json_object_from_text(
    raw_text: str, required_key_list: Optional[Sequence[str]] = None
) -> dict[str, Any]:
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)
    if json_match is not None:
        candidate = json_match.group(1).strip()
    else:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        candidate = raw_text[start : end + 1]
    try:
        info_dict = json.loads(candidate)
    except json.JSONDecodeError:
        info_dict = json.loads(_escape_control_chars_inside_json_strings(candidate))
    if not isinstance(info_dict, dict):
        raise ValueError("Parsed JSON value is not an object.")
    if required_key_list is not None:
        missing_key_list = [key for key in required_key_list if key not in info_dict]
        if missing_key_list:
            raise ValueError(
                f"Missing required JSON keys: {', '.join(missing_key_list)}"
            )
    return info_dict


def _escape_control_chars_inside_json_strings(candidate: str) -> str:
    result_char_list: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                result_char_list.append(char)
                escaped = False
                continue
            if char == "\\":
                result_char_list.append(char)
                escaped = True
                continue
            if char == '"':
                result_char_list.append(char)
                in_string = False
                continue
            if char == "\n":
                result_char_list.append("\\n")
                continue
            if char == "\r":
                result_char_list.append("\\r")
                continue
            if char == "\t":
                result_char_list.append("\\t")
                continue
            result_char_list.append(char)
            continue
        result_char_list.append(char)
        if char == '"':
            in_string = True
            escaped = False
    return "".join(result_char_list)


def coerce_str_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        return [normalized] if normalized else []
    if not isinstance(raw_value, Sequence):
        return []
    result: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            result.append(normalized)
    return result


def normalize_entities(entity_list: Sequence[str]) -> list[str]:
    normalized_entity_set: set[str] = set()
    normalized_entity_list: list[str] = []
    for entity in entity_list:
        normalized = " ".join(tokenize(entity))
        if len(normalized) == 0 or normalized in normalized_entity_set:
            continue
        normalized_entity_set.add(normalized)
        normalized_entity_list.append(entity.strip())
    return normalized_entity_list


def build_trajectory_text(
    session: Session, role_dict: Mapping[Role, str], max_length: int
) -> str:
    if session.chat_history.get_value_length() <= 3:
        return ""
    return truncate_text(
        session.chat_history.get_value_str(
            role_dict,
            start_index=3,
            end_index=None,
        ),
        max_length,
    )


def build_memrl_episode_record(
    session: Session,
    role_dict: Mapping[Role, str],
    max_length: int,
    initial_q_value: float,
    current_step: int,
) -> MemRLEpisodeRecord:
    experience_record: ExperienceRecord = build_session_snippet(
        session,
        role_dict,
        max_length,
    )
    return MemRLEpisodeRecord(
        query=experience_record.query,
        snippet=experience_record.snippet,
        search_text=experience_record.search_text,
        outcome=session.evaluation_record.outcome,
        q_value=initial_q_value,
        retrieve_count=0,
        success_count=0,
        failure_count=0,
        created_step=current_step,
        last_access_step=current_step,
    )


def compute_entity_overlap_score(query: str, entity_list: Sequence[str]) -> float:
    if len(entity_list) == 0:
        return 0.0
    query_token_set = set(tokenize(query))
    if len(query_token_set) == 0:
        return 0.0
    overlap_score = 0.0
    for entity in entity_list:
        entity_token_set = set(tokenize(entity))
        if len(entity_token_set) == 0:
            continue
        if entity_token_set & query_token_set:
            overlap_score += 1.0
    return overlap_score


def rank_mem0_records(
    query: str,
    record_list: Sequence[Mem0Record],
    top_k: int,
    current_step: int,
) -> list[int]:
    if top_k <= 0 or len(record_list) == 0:
        return []
    lexical_rank_index_list = bm25_rank(
        build_retrieval_text(query),
        [record.get_search_text() for record in record_list],
        len(record_list),
    )
    lexical_bonus_dict = {
        record_index: float(len(record_list) - rank)
        for rank, record_index in enumerate(lexical_rank_index_list)
    }
    score_and_index_list: list[tuple[float, int]] = []
    for record_index, record in enumerate(record_list):
        recency_gap = max(0, current_step - record.last_access_step)
        recency_decay = math.exp(-0.015 * recency_gap)
        entity_bonus = compute_entity_overlap_score(query, record.entities)
        score = (
            lexical_bonus_dict.get(record_index, 0.0)
            + 1.5 * record.importance
            + 0.15 * record.mentions
            + 0.75 * entity_bonus
            + recency_decay
        )
        score_and_index_list.append((score, record_index))
    score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [record_index for _, record_index in score_and_index_list[:top_k]]


def rank_memp_records(
    query: str,
    record_list: Sequence[MemPRecord],
    top_k: int,
    current_step: int,
    allowed_kind_set: Optional[set[ProcedureMemoryKind]] = None,
) -> list[int]:
    if top_k <= 0 or len(record_list) == 0:
        return []
    eligible_index_list = [
        index
        for index, record in enumerate(record_list)
        if record.status == ProcedureMemoryStatus.ACTIVE
        and (allowed_kind_set is None or record.kind in allowed_kind_set)
    ]
    if len(eligible_index_list) == 0:
        return []
    relevance_score_dict = {
        index: compute_memp_relevance_score(query, record_list[index])
        for index in eligible_index_list
    }
    if query.strip():
        eligible_index_list = [
            index for index in eligible_index_list if relevance_score_dict[index] > 0
        ]
        if len(eligible_index_list) == 0:
            return []
    eligible_record_list = [record_list[index] for index in eligible_index_list]
    lexical_rank_position_list = bm25_rank(
        query,
        [record.get_search_text() for record in eligible_record_list],
        len(eligible_record_list),
    )
    lexical_bonus_dict = {
        eligible_index_list[position]: float(len(eligible_record_list) - rank)
        for rank, position in enumerate(lexical_rank_position_list)
    }
    score_and_index_list: list[tuple[float, int]] = []
    for record_index in eligible_index_list:
        record = record_list[record_index]
        recency_gap = max(0, current_step - record.last_access_step)
        recency_decay = math.exp(-0.02 * recency_gap)
        score = (
            lexical_bonus_dict.get(record_index, 0.0)
            + 3.0 * relevance_score_dict.get(record_index, 0)
            + 2.0 * record.get_strength()
            + recency_decay
        )
        if record.kind == ProcedureMemoryKind.ABSTRACT:
            score += 0.25
        if record.kind == ProcedureMemoryKind.CAUTION:
            score += 0.1
        score_and_index_list.append((score, record_index))
    score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [record_index for _, record_index in score_and_index_list[:top_k]]


def compute_memp_relevance_score(query: str, record: MemPRecord) -> int:
    query_token_set = _get_retrieval_content_token_set(query)
    record_token_set = _get_retrieval_content_token_set(
        f"{record.source_query}\n{record.memory}"
    )
    content_overlap_count = len(query_token_set & record_token_set)

    if content_overlap_count >= 3:
        return min(content_overlap_count, 4)
    return 0


def _get_retrieval_content_token_set(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in _RETRIEVAL_STOPWORD_SET and len(token) > 2
    }


def select_memrl_candidate_indices(
    query: str,
    record_list: Sequence[MemRLEpisodeRecord],
    top_k: int,
    candidate_count: int,
    current_step: int,
    minimum_relevance_score: int = 2,
) -> list[int]:
    if top_k <= 0 or len(record_list) == 0:
        return []
    relevance_score_dict = {
        index: compute_memrl_relevance_score(query, record)
        for index, record in enumerate(record_list)
    }
    eligible_index_list = [
        index
        for index, relevance_score in relevance_score_dict.items()
        if relevance_score >= minimum_relevance_score
    ]
    if len(eligible_index_list) == 0:
        return []
    lexical_rank_index_list = bm25_rank(
        build_retrieval_text(query),
        [record_list[index].search_text for index in eligible_index_list],
        min(candidate_count, len(eligible_index_list)),
    )
    lexical_rank_index_list = [
        eligible_index_list[index] for index in lexical_rank_index_list
    ]
    lexical_bonus_dict = {
        record_index: float(len(lexical_rank_index_list) - rank)
        for rank, record_index in enumerate(lexical_rank_index_list)
    }
    score_and_index_list: list[tuple[float, int]] = []
    for record_index in lexical_rank_index_list:
        record = record_list[record_index]
        recency_gap = max(0, current_step - record.last_access_step)
        recency_decay = math.exp(-0.02 * recency_gap)
        score = (
            2.5 * record.q_value
            + 0.25 * lexical_bonus_dict.get(record_index, 0.0)
            + 0.05 * record.retrieve_count
            + recency_decay
        )
        score_and_index_list.append((score, record_index))
    score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [record_index for _, record_index in score_and_index_list[:top_k]]


def compute_memrl_relevance_score(query: str, record: MemRLEpisodeRecord) -> int:
    query_token_set = _get_retrieval_content_token_set(build_retrieval_text(query))
    record_token_set = _get_retrieval_content_token_set(record.search_text)
    return len(query_token_set & record_token_set)


def merge_unique_texts(text_list: Sequence[str], max_count: Optional[int] = None) -> list[str]:
    normalized_text_set: set[str] = set()
    merged_text_list: list[str] = []
    for text in text_list:
        normalized = normalize_memory_text(text)
        if len(normalized) == 0 or normalized in normalized_text_set:
            continue
        normalized_text_set.add(normalized)
        merged_text_list.append(text.strip())
        if max_count is not None and len(merged_text_list) >= max_count:
            break
    return merged_text_list


def render_mem0_record(record: Mem0Record) -> str:
    if len(record.entities) == 0:
        return record.memory
    return f"{record.memory} [entities: {', '.join(record.entities)}]"


def render_record_list(record_list: Sequence[MemPRecord]) -> list[str]:
    return [record.get_render_text() for record in record_list]
