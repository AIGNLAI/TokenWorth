"""Isolated hash-cosine retrieval variants for the BigCodeBench methods.

The encoder intentionally matches :class:`TokenWorthMethod`: signed BLAKE2b
hashing over normalized word tokens and 3--5 character n-grams, followed by L2
normalization.  These subclasses change retrieval only; their parent classes
still own memory construction, feedback, capacity management, and rendering.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Sequence

from .evosc import EvoSCMethod
from .evoskill import EvoSkillMethod
from .mem0 import Mem0Method
from .memp import MemPMethod
from .memrl import MemRLMethod
from .memskill import MemSkillMethod
from .rag import RAGMethod
from .reflexion import ReflexionMethod
from .core import RetrievedItem, cosine, tokenize, truncate


HASH_COSINE_VARIANT = "tokenworth_hash_cosine"


def tokenworth_hash_encode(
    text: str,
    *,
    dimension: int = 512,
    character_ngram_min: int = 3,
    character_ngram_max: int = 5,
) -> list[float]:
    """Encode text exactly like TokenWorth's lightweight hashing encoder."""
    if dimension < 1:
        raise ValueError("embedding_dim must be positive")
    if character_ngram_min < 1 or character_ngram_max < character_ngram_min:
        raise ValueError("invalid character n-gram range")
    vector = [0.0] * dimension
    tokens = tokenize(text)
    normalized_text = " ".join(tokens)

    def add(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if int.from_bytes(digest[4:], "big") % 2 == 0 else -1.0
        vector[bucket] += sign * weight

    for token in tokens:
        add(f"tok:{token}", 1.0)
    for size in range(character_ngram_min, character_ngram_max + 1):
        if len(normalized_text) < size:
            continue
        weight = 0.35 / math.sqrt(size)
        for index in range(len(normalized_text) - size + 1):
            add(f"chr{size}:{normalized_text[index:index + size]}", weight)
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 1e-12 else vector


class HashCosineRetrievalMixin:
    """Shared hash embeddings and cosine relevance for native rankers."""

    def _hash_encode(self, text: str) -> list[float]:
        return tokenworth_hash_encode(
            text,
            dimension=int(self.config.get("embedding_dim", 512)),
            character_ngram_min=int(self.config.get("character_ngram_min", 3)),
            character_ngram_max=int(self.config.get("character_ngram_max", 5)),
        )

    def _embedding_key(self, text: str) -> str:
        signature = (
            f"{int(self.config.get('embedding_dim', 512))}:"
            f"{int(self.config.get('character_ngram_min', 3))}:"
            f"{int(self.config.get('character_ngram_max', 5))}:"
        )
        return signature + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _record_embedding(self, record: dict[str, Any]) -> list[float]:
        text = str(record.get("search_text", ""))
        key = self._embedding_key(text)
        if record.get("hash_embedding_key") != key:
            record["hash_embedding"] = self._hash_encode(text)
            record["hash_embedding_key"] = key
        return [float(value) for value in record.get("hash_embedding", [])]

    def _cosine_rank(
        self,
        prompt: str,
        records: Sequence[dict[str, Any]],
        eligible: Sequence[int] | None = None,
    ) -> list[tuple[float, int]]:
        threshold = self.retrieval_threshold(
            "min_retrieval_similarity", 0.05
        )
        ranked = self._cosine_scores(prompt, records, eligible)
        ranked = [item for item in ranked if item[0] >= threshold]
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return ranked

    def _cosine_scores(
        self,
        prompt: str,
        records: Sequence[dict[str, Any]],
        eligible: Sequence[int] | None = None,
    ) -> list[tuple[float, int]]:
        query = self._hash_encode(prompt)
        indices = list(range(len(records))) if eligible is None else list(eligible)
        return [
            (cosine(query, self._record_embedding(records[index])), index)
            for index in indices
        ]

    def summary(self) -> dict[str, Any]:
        return {
            **super().summary(),
            "retrieval_variant": HASH_COSINE_VARIANT,
            "embedding_dim": int(self.config.get("embedding_dim", 512)),
            "character_ngram_range": [
                int(self.config.get("character_ngram_min", 3)),
                int(self.config.get("character_ngram_max", 5)),
            ],
            "base_similarity": "cosine",
            "ranking": "method_native_hybrid",
        }


class RAGHashCosineMethod(HashCosineRetrievalMixin, RAGMethod):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("experiences", [])
        selected = self._cosine_rank(prompt, records)[: int(self.config.get("top_k", 4))]
        self.last_retrieved = [index for _, index in selected]
        return [
            RetrievedItem.from_record(
                records[index], score, records[index]["snippet"],
                include_method_memory=False,
            )
            for score, index in selected
        ]


class Mem0HashCosineMethod(HashCosineRetrievalMixin, Mem0Method):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("memories", [])
        eligible = [
            index for index, record in enumerate(records)
            if record["outcome"] == "pass"
        ] or list(range(len(records)))
        threshold = self.retrieval_threshold(
            "min_retrieval_similarity", 0.05
        )
        scored: list[tuple[float, int]] = []
        for similarity, index in self._cosine_scores(prompt, records, eligible):
            if similarity < threshold:
                continue
            record = records[index]
            recency = math.exp(
                -0.015 * max(0, self.step - record["last_access_step"])
            )
            score = (
                similarity
                + 1.5 * record["importance"]
                + 0.15 * record["mentions"]
                + recency
            )
            scored.append((score, index))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = scored[: int(self.config.get("top_k", 8))]
        self.last_retrieved = [index for _, index in selected]
        for _, index in selected:
            records[index]["last_access_step"] = self.step
            records[index]["mentions"] += 1
            records[index]["importance"] += 0.05
        return [
            RetrievedItem.from_record(records[index], score, records[index]["memory"])
            for score, index in selected
        ]


class MemPHashCosineMethod(HashCosineRetrievalMixin, MemPMethod):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("procedures", [])
        eligible = [
            index for index, record in enumerate(records)
            if record["status"] == "active"
        ]
        threshold = self.retrieval_threshold(
            "min_retrieval_similarity", 0.05
        )
        ranked = sorted(
            (
                (
                    similarity
                    + 2.0
                    * (
                        records[index]["support_count"]
                        - 0.75 * records[index]["contradiction_count"]
                    ),
                    index,
                )
                for similarity, index in self._cosine_scores(
                    prompt, records, eligible
                )
                if similarity >= threshold
            ),
            reverse=True,
        )
        selected: list[tuple[float, int]] = []
        for kind, limit in (("abstract", 1), ("detailed", 2), ("caution", 1)):
            selected.extend(
                [
                    item
                    for item in ranked
                    if records[item[1]]["kind"] == kind
                ][:limit]
            )
        # Preserve MemP's type diversity, then fill by the same hybrid score.
        deduplicated: list[tuple[float, int]] = []
        seen: set[int] = set()
        for item in selected + ranked:
            if item[1] not in seen:
                deduplicated.append(item)
                seen.add(item[1])
        selected = deduplicated[: min(int(self.config.get("top_k", 8)), 4)]
        self.last_retrieved = [index for _, index in selected]
        return [
            RetrievedItem.from_record(
                records[index], score,
                f"[{records[index]['kind']}] {records[index]['memory']}",
            )
            for score, index in selected
        ]


class MemRLHashCosineMethod(HashCosineRetrievalMixin, MemRLMethod):
    def _similarity(self, left: str, right: str) -> float:
        return cosine(self._hash_encode(left), self._hash_encode(right))

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("episodes", [])
        threshold = self.retrieval_threshold("similarity_threshold", 0.20)
        candidates = [
            (similarity, index)
            for similarity, index in self._cosine_scores(prompt, records)
            if similarity >= threshold
        ]
        candidates.sort(reverse=True)
        candidates = candidates[: int(self.config.get("candidate_count", 16))]

        similarity_weight = float(self.config.get("weight_similarity", 0.5))
        q_weight = float(self.config.get("weight_q", 0.5))
        denominator = similarity_weight + q_weight
        if denominator <= 0:
            similarity_weight, q_weight, denominator = 1.0, 0.0, 1.0
        ranked = [
            (
                (
                    similarity_weight * similarity
                    + q_weight
                    * self._q_score(float(records[index]["q_value"]))
                )
                / denominator,
                index,
            )
            for similarity, index in candidates
        ]
        ranked.sort(reverse=True)

        top_k = int(self.config.get("top_k", 5))
        epsilon = float(self.config.get("epsilon", 0.0))
        rng = random.Random(int(self.config.get("random_seed", 42)) + self.step)
        if ranked and epsilon > 0 and rng.random() < epsilon:
            selected = rng.sample(ranked, min(top_k, len(ranked)))
            selected.sort(reverse=True)
        else:
            selected = ranked[:top_k]
        self.last_retrieved = [index for _, index in selected]
        for _, index in selected:
            records[index]["retrieve_count"] += 1
            records[index]["last_access_step"] = self.step
        return [
            RetrievedItem.from_record(records[index], score, records[index]["snippet"])
            for score, index in selected
        ]


class MemSkillHashCosineMethod(HashCosineRetrievalMixin, MemSkillMethod):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state["memories"]
        eligible = [
            index for index, record in enumerate(records)
            if not self.config.get("correct_memory_only", True)
            or record["source_outcome"] == "pass"
        ]
        threshold = self.retrieval_threshold(
            "min_retrieval_similarity", 0.05
        )
        ranked = sorted(
            (
                (
                    similarity
                    + 0.5 * records[index]["strength"]
                    + 1.0
                    / (self.step - records[index]["last_access_step"] + 1),
                    index,
                )
                for similarity, index in self._cosine_scores(
                    prompt, records, eligible
                )
                if similarity >= threshold
            ),
            reverse=True,
        )
        selected = ranked[: int(self.config.get("top_k", 8))]
        self.last_retrieved = [index for _, index in selected]
        for _, index in selected:
            records[index]["access_count"] += 1
            records[index]["last_access_step"] = self.step
        return [
            RetrievedItem.from_record(
                records[index], score,
                truncate(
                    records[index]["memory"],
                    int(self.config.get("max_injected_memory_chars_per_item", 1500)),
                ),
            )
            for score, index in selected
        ]

    def _rank_skills(
        self,
        query: str,
        skills: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[int]:
        records = [
            {
                "search_text": str(skill.get("description", "")),
            }
            for skill in skills
        ]
        ranked = self._cosine_rank(query, records)[:top_k]
        return [index for _, index in ranked]


class ReflexionHashCosineMethod(HashCosineRetrievalMixin, ReflexionMethod):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("reflections", [])
        selected = self._cosine_rank(prompt, records)[
            : int(self.config.get("top_k", 4))
        ]
        self.last_retrieved = [index for _, index in selected]
        return [
            RetrievedItem.from_record(
                records[index], score, records[index]["reflection"]
            )
            for score, index in selected
        ]


class EvoSkillHashCosineMethod(HashCosineRetrievalMixin, EvoSkillMethod):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        skills = self.state["skills"]
        eligible = [
            index for index, skill in enumerate(skills)
            if skill["status"] in {"active", "candidate"}
        ]
        threshold = self.retrieval_threshold(
            "min_retrieval_similarity", 0.05
        )
        ranked = sorted(
            (
                (score, index)
                for score, index in self._cosine_scores(prompt, skills, eligible)
                if score >= threshold
            ),
            reverse=True,
        )
        selected = ranked[: int(self.config.get("top_k", 4))]
        self.last_retrieved = [index for _, index in selected]
        return [
            RetrievedItem.from_record(
                skills[index], score,
                f"Skill: {skills[index]['name']}\n"
                f"Triggers: {'; '.join(skills[index]['triggers'])}\n"
                f"Procedure:\n{skills[index]['instruction']}",
            )
            for score, index in selected
        ]


class EvoSCHashCosineMethod(HashCosineRetrievalMixin, EvoSCMethod):
    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        limits = {
            "summary": 2,
            "success": min(2, int(self.config.get("top_k_success", 4))),
            "failure": min(2, int(self.config.get("top_k_failure", 4))),
        }
        selected: list[RetrievedItem] = []
        for kind, state_key in (
            ("summary", "summaries"),
            ("success", "successes"),
            ("failure", "failures"),
        ):
            records = self.state[state_key]
            threshold = self.retrieval_threshold(
                "min_retrieval_similarity", 0.05
            )
            ranked = sorted(
                (
                    item
                    for item in self._cosine_scores(prompt, records)
                    if item[0] >= threshold
                ),
                reverse=True,
            )[: limits[kind]]
            selected.extend(
                RetrievedItem.from_record(
                    records[index], score, f"[{kind}] {records[index]['content']}"
                )
                for score, index in ranked
            )
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected[: int(self.config.get("top_k", 5))]


HASH_COSINE_METHODS = {
    "RAG": RAGHashCosineMethod,
    "Mem0": Mem0HashCosineMethod,
    "MemP": MemPHashCosineMethod,
    "MemRL": MemRLHashCosineMethod,
    "MemSkill": MemSkillHashCosineMethod,
    "Reflexion": ReflexionHashCosineMethod,
    "EvoSkill": EvoSkillHashCosineMethod,
    "EvoSC": EvoSCHashCosineMethod,
}
