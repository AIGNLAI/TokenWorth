import json
import math
import os
import re
from collections import Counter
from typing import Mapping, Optional, Sequence, TypeVar

from pydantic import BaseModel

from src.language_models import LanguageModel
from src.utils import SafeLogger
from src.typings import (
    ChatHistory,
    ChatHistoryItem,
    Role,
    Session,
    SessionEvaluationOutcome,
    GenerationBudgetExceeded,
)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")


class ExperienceRecord(BaseModel):
    query: str
    snippet: str
    search_text: str
    outcome: SessionEvaluationOutcome


class ReflectionRecord(BaseModel):
    query: str
    reflection: str
    search_text: str


class MemoryBankRecord(BaseModel):
    memory: str
    source_query: str
    outcome: SessionEvaluationOutcome
    created_step: int
    last_access_step: int
    access_count: int
    strength: float

    def get_search_text(self) -> str:
        return f"{self.source_query}\n{self.memory}"


ModelType = TypeVar("ModelType", bound=BaseModel)


def load_json_state_list(
    state_path: str, model_cls: type[ModelType]
) -> list[ModelType]:
    if not os.path.exists(state_path):
        return []
    return [
        model_cls.model_validate(item_dict)
        for item_dict in json.load(open(state_path, "r"))
    ]


def dump_json_state_list(state_path: str, state_list: Sequence[BaseModel]) -> None:
    json.dump(
        [state.model_dump() for state in state_list],
        open(state_path, "w"),  # noqa
        indent=2,
    )


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def build_retrieval_text(text: str) -> str:
    return text


def rank_texts_by_relevance(
    query: str,
    document_text_list: Sequence[str],
    top_k: int,
) -> list[int]:
    return bm25_rank(
        build_retrieval_text(query),
        [build_retrieval_text(document_text) for document_text in document_text_list],
        top_k,
    )


def bm25_rank(
    query: str,
    document_text_list: Sequence[str],
    top_k: int,
) -> list[int]:
    if top_k <= 0 or len(document_text_list) == 0:
        return []
    score_list = bm25_scores(query, document_text_list)
    score_and_index_list = list(enumerate(score_list))
    score_and_index_list = [
        (score, document_index) for document_index, score in score_and_index_list
    ]
    score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    positive_score_index_list = [
        document_index for score, document_index in score_and_index_list if score > 0
    ]
    if len(positive_score_index_list) >= top_k:
        return positive_score_index_list[:top_k]
    fallback_index_list = [
        document_index
        for _, document_index in score_and_index_list
        if document_index not in positive_score_index_list
    ]
    return (positive_score_index_list + fallback_index_list)[:top_k]


def bm25_scores(query: str, document_text_list: Sequence[str]) -> list[float]:
    """Return BM25 scores in input document order."""
    if len(document_text_list) == 0:
        return []
    query_token_list = tokenize(query)
    if len(query_token_list) == 0:
        return [0.0 for _ in document_text_list]
    document_token_counter_list = [
        Counter(tokenize(document_text)) for document_text in document_text_list
    ]
    document_length_list = [
        sum(token_counter.values()) for token_counter in document_token_counter_list
    ]
    average_document_length = (
        sum(document_length_list) / len(document_length_list)
        if len(document_length_list) > 0
        else 0.0
    )
    document_frequency_dict: Counter[str] = Counter()
    for token_counter in document_token_counter_list:
        for token in token_counter.keys():
            document_frequency_dict[token] += 1
    score_list: list[float] = []
    k1 = 1.5
    b = 0.75
    for document_index, token_counter in enumerate(document_token_counter_list):
        score = 0.0
        document_length = document_length_list[document_index]
        for token in query_token_list:
            term_frequency = token_counter.get(token, 0)
            if term_frequency == 0:
                continue
            document_frequency = document_frequency_dict[token]
            idf = math.log(
                1.0
                + (len(document_text_list) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1.0
                - b
                + b
                * (
                    document_length / average_document_length
                    if average_document_length > 0
                    else 0.0
                )
            )
            score += idf * term_frequency * (k1 + 1.0) / denominator
        score_list.append(score)
    return score_list


def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3] + "..."


def truncate_text_list_total(text_list: Sequence[str], max_total_length: int) -> list[str]:
    result: list[str] = []
    remaining_length = max_total_length
    for text in text_list:
        if remaining_length <= 0:
            break
        truncated_text = truncate_text(text, remaining_length).strip()
        if len(truncated_text) == 0:
            continue
        result.append(truncated_text)
        remaining_length -= len(truncated_text)
    return result


def extract_question_from_session(session: Session) -> str:
    if session.chat_history.get_value_length() < 3:
        return ""
    return session.chat_history.get_item_deep_copy(2).content


def build_session_snippet(
    session: Session, role_dict: Mapping[Role, str], max_length: int
) -> ExperienceRecord:
    query = extract_question_from_session(session)
    if session.chat_history.get_value_length() > 3:
        conversation = session.chat_history.get_value_str(
            role_dict,
            start_index=3,
            end_index=None,
        )
    else:
        conversation = ""
    snippet = f"Past question:\n{query}\n\nPast trajectory:\n{truncate_text(conversation, max_length)}"
    return ExperienceRecord(
        query=query,
        snippet=snippet,
        search_text=build_retrieval_text(f"{query}\n{conversation}"),
        outcome=session.evaluation_record.outcome,
    )


def inject_memory_block(
    chat_history: ChatHistory,
    title: str,
    bullet_text_list: Sequence[str],
) -> None:
    if len(bullet_text_list) == 0:
        return
    original_first_item = chat_history.get_item_deep_copy(0)
    assert original_first_item.role == Role.USER
    memory_block = "\n".join([f"- {bullet_text}" for bullet_text in bullet_text_list])
    memory_block = f"{title}:\n{memory_block}"
    anchor = "\n\nNow, I will give you the question that you need to solve."
    if anchor in original_first_item.content:
        updated_content = original_first_item.content.replace(
            anchor,
            f"\n\n{memory_block}{anchor}",
            1,
        )
    else:
        updated_content = f"{original_first_item.content}\n\n{memory_block}"
    chat_history.set(
        0,
        ChatHistoryItem(
            role=Role.USER,
            content=updated_content,
        ),
    )


def run_single_prompt(
    language_model: LanguageModel,
    prompt: str,
    *,
    max_tokens: int | None = None,
) -> str:
    chat_history = ChatHistory()
    chat_history.inject(ChatHistoryItem(role=Role.USER, content=prompt))
    inference_config_dict = None
    if max_tokens is not None:
        inference_config_dict = {
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1.0,
        }
    SafeLogger.info(
        "[AuxiliaryMemoryInference] "
        f"max_tokens={max_tokens if max_tokens is not None else 'default'}"
    )
    language_model.set_usage_context(phase="auxiliary")
    try:
        return language_model.inference(
            [chat_history], inference_config_dict=inference_config_dict
        )[0].content.strip()
    except GenerationBudgetExceeded:
        # Auxiliary generation cannot borrow tokens from the next task.
        return ""
    finally:
        language_model.set_usage_context(phase="agent")


def parse_bullet_list(raw_text: str, max_item_count: int) -> list[str]:
    if max_item_count <= 0:
        return []
    result: list[str] = []
    for line in raw_text.splitlines():
        match = _LIST_ITEM_PATTERN.match(line)
        if match is None:
            continue
        content = match.group(1).strip()
        if len(content) == 0:
            continue
        result.append(content)
        if len(result) >= max_item_count:
            return result
    fallback_text = raw_text.strip()
    if len(result) == 0 and len(fallback_text) > 0:
        return [fallback_text]
    return result


def normalize_memory_text(text: str) -> str:
    return " ".join(tokenize(text))


def rank_memory_bank_records(
    query: str,
    memory_bank_record_list: Sequence[MemoryBankRecord],
    top_k: int,
    current_step: int,
) -> list[int]:
    if top_k <= 0 or len(memory_bank_record_list) == 0:
        return []
    lexical_rank_index_list = bm25_rank(
        build_retrieval_text(query),
        [
            build_retrieval_text(record.get_search_text())
            for record in memory_bank_record_list
        ],
        len(memory_bank_record_list),
    )
    lexical_bonus_dict = {
        memory_index: float(len(memory_bank_record_list) - rank)
        for rank, memory_index in enumerate(lexical_rank_index_list)
    }
    score_and_index_list: list[tuple[float, int]] = []
    for memory_index, record in enumerate(memory_bank_record_list):
        recency_gap = max(0, current_step - record.last_access_step)
        recency_decay = math.exp(-0.02 * recency_gap)
        score = (
            lexical_bonus_dict.get(memory_index, 0.0)
            + 2.0 * record.strength
            + 0.2 * record.access_count
            + recency_decay
        )
        score_and_index_list.append((score, memory_index))
    score_and_index_list.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [memory_index for _, memory_index in score_and_index_list[:top_k]]
