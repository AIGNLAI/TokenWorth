"""Dependency-free retrieval shared by every standardized baseline."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokens(text: str) -> List[str]:
    raw = [token.lower() for token in TOKEN_RE.findall(text)]
    return raw + [raw[index] + "::" + raw[index + 1] for index in range(len(raw) - 1)]


def hashed_vector(text: str, dimensions: int = 512) -> Dict[int, float]:
    """Return a stable signed feature-hashing vector.

    ALFWorld objectives have highly repetitive lexical templates, so this is a
    useful deterministic default when the chat endpoint has no embeddings API.
    Every method uses the same representation, preventing an embedding model
    from becoming an uncontrolled method-specific variable.
    """
    vector: Dict[int, float] = {}
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = 1.0 if value & (1 << 63) else -1.0
        vector[index] = vector.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        vector = {index: value / norm for index, value in vector.items()}
    return vector


def tokenworth_hashed_vector(
    text: str,
    dimensions: int = 512,
    character_ngram_min: int = 3,
    character_ngram_max: int = 5,
) -> List[float]:
    """Return the dense token/character hashing representation used by TokenWorth."""

    vector = [0.0 for _ in range(dimensions)]

    def add_feature(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if int.from_bytes(digest[4:], "big") % 2 == 0 else -1.0
        vector[bucket] += sign * weight

    words = normalize(text).split()
    for token in words:
        add_feature("tok:" + token, 1.0)
    normalized_text = " ".join(words)
    for size in range(character_ngram_min, character_ngram_max + 1):
        if len(normalized_text) < size:
            continue
        weight = 0.35 / math.sqrt(size)
        for index in range(len(normalized_text) - size + 1):
            add_feature(
                "chr{}:{}".format(size, normalized_text[index : index + size]),
                weight,
            )
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 1e-12 else vector


def tokenworth_hash_rank(
    query: str,
    documents: Sequence[str],
    top_k: int,
    minimum_score: float = -1.0,
    dimensions: int = 512,
    character_ngram_min: int = 3,
    character_ngram_max: int = 5,
) -> List[Tuple[float, int]]:
    """Rank texts with TokenWorth's normalized token/character Hash encoder."""

    query_vector = tokenworth_hashed_vector(
        query,
        dimensions=dimensions,
        character_ngram_min=character_ngram_min,
        character_ngram_max=character_ngram_max,
    )
    scored = []
    for index, document in enumerate(documents):
        document_vector = tokenworth_hashed_vector(
            document,
            dimensions=dimensions,
            character_ngram_min=character_ngram_min,
            character_ngram_max=character_ngram_max,
        )
        # Both vectors are L2-normalized, so their dot product is cosine.
        score = sum(left * right for left, right in zip(query_vector, document_vector))
        if score >= minimum_score:
            scored.append((score, index))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored[: max(0, top_k)]


def cosine(left: Dict[int, float], right: Dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def rank(
    query: str,
    documents: Sequence[str],
    top_k: int,
    minimum_score: float = -1.0,
    dimensions: int = 512,
) -> List[Tuple[float, int]]:
    query_vector = hashed_vector(query, dimensions)
    scored = [
        (cosine(query_vector, hashed_vector(document, dimensions)), index)
        for index, document in enumerate(documents)
    ]
    scored = [item for item in scored if item[0] >= minimum_score]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored[: max(0, top_k)]


def bm25_scores(query: str, documents: Sequence[str]) -> List[float]:
    """BM25 used by TokenWorth's RAG callback, kept dependency-free."""
    if not documents:
        return []
    query_tokens = [token.lower() for token in TOKEN_RE.findall(query)]
    if not query_tokens:
        return [0.0 for _ in documents]
    counters = [Counter(token.lower() for token in TOKEN_RE.findall(document)) for document in documents]
    lengths = [sum(counter.values()) for counter in counters]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    frequencies: Counter = Counter()
    for counter in counters:
        for token in counter:
            frequencies[token] += 1
    scores: List[float] = []
    k1, b = 1.5, 0.75
    for index, counter in enumerate(counters):
        score = 0.0
        for token in query_tokens:
            term_frequency = counter.get(token, 0)
            if not term_frequency:
                continue
            document_frequency = frequencies[token]
            inverse_frequency = math.log(
                1.0
                + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1.0
                - b
                + b * (lengths[index] / average_length if average_length else 0.0)
            )
            score += inverse_frequency * term_frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


def bm25_rank(query: str, documents: Sequence[str], top_k: int) -> List[Tuple[float, int]]:
    scored = [(score, index) for index, score in enumerate(bm25_scores(query, documents))]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored[: max(0, top_k)]


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def unique_strings(values: Iterable[object], limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        key = normalize(value)
        if key and key not in seen:
            result.append(value.strip())
            seen.add(key)
        if len(result) >= limit:
            break
    return result
