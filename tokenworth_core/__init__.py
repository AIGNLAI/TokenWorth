"""Online TokenWorth memory retrieval and learning."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

VERSION = "paper-v2"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
PRIOR = [1.0, 0.10, 0.10, 0.02]
DEFAULTS = {
    "embedding_dim": 512,
    "character_ngram_min": 3,
    "character_ngram_max": 5,
    "top_k": 1,
    "candidate_count": 12,
    "capacity": 64,
    "krr_update_interval": 75,
    "agop_update_interval": 225,
    "krr_buffer_size": 128,
    "agop_buffer_size": 256,
    "min_pair_count_for_rfm": 4,
    "ridge_lambda": 0.01,
    "kernel_gamma": 1.0,
    "agop_smoothing": 0.25,
    "agop_min_weight": 0.05,
    "agop_max_weight": 8.0,
    "rfm_influence": 0.10,
    "rfm_influence_warmup_updates": 3,
    "base_buffer_size": 256,
    "base_update_interval": 50,
    "base_regularization": 0.10,
    "base_smoothing": 0.50,
    "novelty_threshold": 0.35,
    "recent_probe_count": 8,
    "write_gain_margin": 0.02,
    "positive_ratio_threshold": 0.45,
    "replacement_value_margin": 0.10,
}


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("TokenWorth embedding dimensions do not match")
    denom = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(a * b for a, b in zip(left, right)) / denom if denom > 1e-12 else 0.0


def encode(
    text: str, dimensions: int = 512, ngram_min: int = 3, ngram_max: int = 5
) -> List[float]:
    vector = [0.0] * dimensions
    words = TOKEN_RE.findall(text.lower())

    def add(feature, weight):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if int.from_bytes(digest[4:], "big") % 2 == 0 else -1.0
        vector[index] += sign * weight

    for word in words:
        add("tok:" + word, 1.0)
    normalized = " ".join(words)
    for size in range(ngram_min, ngram_max + 1):
        for index in range(max(0, len(normalized) - size + 1)):
            add(
                "chr{}:{}".format(size, normalized[index : index + size]),
                0.35 / math.sqrt(size),
            )
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 1e-12 else vector


def shortlist(query: str, documents: Sequence[str], count: int) -> List[int]:
    counters = [Counter(TOKEN_RE.findall(text.lower())) for text in documents]
    lengths = [sum(c.values()) for c in counters]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    frequencies = Counter()
    for counter in counters:
        frequencies.update(counter.keys())
    scores = []
    for index, (counter, length) in enumerate(zip(counters, lengths)):
        score = 0.0
        for word in TOKEN_RE.findall(query.lower()):
            frequency = counter.get(word, 0)
            if frequency:
                idf = math.log(
                    1.0
                    + (len(documents) - frequencies[word] + 0.5)
                    / (frequencies[word] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * (length / average if average else 0.0)
                )
                score += idf * frequency * 2.5 / denominator
        scores.append((score, index))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in scores[:count]]


class TokenWorthCore:
    """Selected-memory feedback, learned base, residual KRR/AGOP and bank control.

    Retrieval only caches features. Accesses and feedback are committed after a
    valid external outcome, using exactly the IDs whose snippets were injected.
    This makes failed model/evaluator attempts incapable of becoming feedback.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        supplied = parameters or {}
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                "Unknown paper TokenWorth parameters: " + ", ".join(sorted(unknown))
            )
        self.parameters = {**DEFAULTS, **supplied}
        p = self.parameters
        for key in (
            "embedding_dim",
            "top_k",
            "candidate_count",
            "capacity",
            "krr_update_interval",
            "agop_update_interval",
            "krr_buffer_size",
            "agop_buffer_size",
            "base_buffer_size",
            "base_update_interval",
            "min_pair_count_for_rfm",
            "rfm_influence_warmup_updates",
            "character_ngram_min",
            "character_ngram_max",
        ):
            if isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < 1:
                raise ValueError("{} must be a positive integer".format(key))
        if (
            p["candidate_count"] < p["top_k"]
            or p["agop_buffer_size"] < p["krr_buffer_size"]
        ):
            raise ValueError("Invalid candidate/buffer capacities")
        if p["character_ngram_max"] < p["character_ngram_min"]:
            raise ValueError("Invalid character ngram range")
        if p["min_pair_count_for_rfm"] > p["krr_buffer_size"]:
            raise ValueError("Residual minimum exceeds fast buffer")
        for key in (
            "ridge_lambda",
            "kernel_gamma",
            "base_regularization",
            "agop_min_weight",
            "agop_max_weight",
        ):
            if not math.isfinite(p[key]) or p[key] <= 0:
                raise ValueError("{} must be finite and positive".format(key))
        for key in ("agop_smoothing", "base_smoothing", "positive_ratio_threshold"):
            if not math.isfinite(p[key]) or not 0 <= p[key] <= 1:
                raise ValueError("{} must be in [0,1]".format(key))
        for key in (
            "rfm_influence",
            "novelty_threshold",
            "write_gain_margin",
            "replacement_value_margin",
        ):
            if not math.isfinite(p[key]) or p[key] < 0:
                raise ValueError("{} must be finite and nonnegative".format(key))
        if p["agop_min_weight"] > p["agop_max_weight"]:
            raise ValueError("Invalid AGOP clipping interval")
        if (
            isinstance(p["recent_probe_count"], bool)
            or not isinstance(p["recent_probe_count"], int)
            or p["recent_probe_count"] < 0
        ):
            raise ValueError("recent_probe_count must be nonnegative")
        self.state = (
            state
            if state is not None
            else {
                "version": VERSION,
                "parameters": deepcopy(p),
                "step": 0,
                "memories": [],
                "next_id": 0,
                "base_weights": list(PRIOR),
                "base_pairs": [],
                "pairs": [],
                "agop_pairs": [],
                "krr_train": [],
                "kernel_alpha": [],
                "agop_diag": [1.0] * (p["embedding_dim"] + 5),
                "recent_queries": [],
                "metrics": {
                    "pair_writes": 0,
                    "positive_pair_count": 0,
                    "negative_pair_count": 0,
                    "base_updates": 0,
                    "krr_updates": 0,
                    "agop_updates": 0,
                    "write_count": 0,
                    "replace_count": 0,
                    "evict_count": 0,
                    "write_skip_count": 0,
                    "retrieve_count": 0,
                },
            }
        )
        self.validate_state()
        self._predictor_source = None
        self._predictor_arrays = None
        self._query = None
        self._query_embedding = []
        self._selected = {}
        self._injected = set()

    def validate_state(self):
        s, p = self.state, self.parameters
        if s.get("version") != VERSION or s.get("parameters") != p:
            raise ValueError(
                "Incompatible TokenWorth state/configuration; use a new output directory"
            )
        required = {
            "step",
            "memories",
            "next_id",
            "base_weights",
            "base_pairs",
            "pairs",
            "agop_pairs",
            "krr_train",
            "kernel_alpha",
            "agop_diag",
            "recent_queries",
            "metrics",
        }
        if not required.issubset(s):
            raise ValueError("Incomplete paper TokenWorth checkpoint")
        dimension = p["embedding_dim"] + 5

        def finite_vector(vector, length):
            return (
                isinstance(vector, list)
                and len(vector) == length
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in vector
                )
            )

        if not finite_vector(s["base_weights"], 4) or any(
            not 0 <= x <= bound for x, bound in zip(s["base_weights"], [2, 1, 1, 1])
        ):
            raise ValueError("Invalid learned base weights")
        if not finite_vector(s["agop_diag"], dimension) or min(s["agop_diag"]) <= 0:
            raise ValueError("Invalid AGOP metric")
        if not math.isclose(sum(s["agop_diag"]) / dimension, 1.0, rel_tol=1e-8):
            raise ValueError("AGOP metric must have mean one")
        for name, capacity, width in (
            ("base_pairs", p["base_buffer_size"], 4),
            ("pairs", p["krr_buffer_size"], dimension),
            ("agop_pairs", p["agop_buffer_size"], dimension),
            ("krr_train", p["krr_buffer_size"], dimension),
        ):
            if len(s[name]) > capacity:
                raise ValueError("TokenWorth feedback buffer exceeds its capacity")
            for pair in s[name]:
                if not finite_vector(pair.get("feature"), width) or not math.isfinite(
                    pair.get("label", float("nan"))
                ):
                    raise ValueError("Invalid TokenWorth feedback row")
        if not finite_vector(s["kernel_alpha"], len(s["krr_train"])):
            raise ValueError("KRR coefficients/training rows mismatch")
        if len(s["memories"]) > p["capacity"]:
            raise ValueError("TokenWorth bank exceeds its capacity")
        ids = []
        for record in s["memories"]:
            ids.append(record["id"])
            if not finite_vector(
                record.get("embedding"), p["embedding_dim"]
            ) or not record.get("snippet"):
                raise ValueError("Invalid TokenWorth memory")
            for key in (
                "created_step",
                "last_access_step",
                "access_count",
                "help_count",
                "harm_count",
            ):
                if not isinstance(record.get(key), int) or record[key] < 0:
                    raise ValueError("Invalid memory counters")
            if (
                not math.isfinite(record.get("utility", float("nan")))
                or not -2 <= record["utility"] <= 5
            ):
                raise ValueError("Invalid memory utility")
        if len(set(ids)) != len(ids) or any(
            not isinstance(x, int) or x < 0 or x >= s["next_id"] for x in ids
        ):
            raise ValueError("Invalid TokenWorth memory IDs")
        if (
            not isinstance(s["step"], int)
            or s["step"] < 0
            or not isinstance(s["next_id"], int)
            or s["next_id"] < 0
        ):
            raise ValueError("Invalid TokenWorth step/ID counter")
        if len(s["recent_queries"]) > p["recent_probe_count"] or any(
            not finite_vector(x, p["embedding_dim"]) for x in s["recent_queries"]
        ):
            raise ValueError("Invalid query probe history")
        for key in (
            "pair_writes",
            "positive_pair_count",
            "negative_pair_count",
            "base_updates",
            "krr_updates",
            "agop_updates",
            "write_count",
            "replace_count",
            "evict_count",
            "write_skip_count",
            "retrieve_count",
        ):
            if not isinstance(s["metrics"].get(key), int) or s["metrics"][key] < 0:
                raise ValueError("Invalid TokenWorth learning counters")
        if (
            s["metrics"]["positive_pair_count"] + s["metrics"]["negative_pair_count"]
            != s["metrics"]["pair_writes"]
        ):
            raise ValueError("Inconsistent feedback counters")

    def encode(self, text):
        p = self.parameters
        return encode(
            text, p["embedding_dim"], p["character_ngram_min"], p["character_ngram_max"]
        )

    def feature(self, embedding, record, step=None):
        t = self.state["step"] + 1 if step is None else step
        age = max(1, t - record["created_step"] + 1)
        return [abs(a - b) for a, b in zip(embedding, record["embedding"])] + [
            cosine(embedding, record["embedding"]),
            record["utility"],
            (record["help_count"] + 1.0)
            / (record["help_count"] + record["harm_count"] + 2.0),
            math.exp(-0.02 * max(0, t - record["last_access_step"])),
            record["access_count"] / age,
        ]

    @staticmethod
    def base_features(feature):
        return list(feature[-5:-1])

    def base_score(self, feature):
        return sum(
            w * x
            for w, x in zip(self.state["base_weights"], self.base_features(feature))
        )

    def kernel(self, left, right):
        d = sum(
            w * (a - b) ** 2 for w, a, b in zip(self.state["agop_diag"], left, right)
        )
        return math.exp(-self.parameters["kernel_gamma"] * math.sqrt(max(0.0, d)))

    def residual(self, feature):
        s = self.state
        if not s["kernel_alpha"]:
            return 0.0
        source = (s["krr_train"], s["kernel_alpha"], s["agop_diag"])
        # Fitted rows, coefficients and metric are replaced atomically on a
        # fit/update. Reuse their numeric representation across all bank probes;
        # never retain a stale geometry after an AGOP update or resumed state.
        if self._predictor_source is None or any(
            current is not previous
            for current, previous in zip(source, self._predictor_source)
        ):
            self._predictor_source = source
            self._predictor_arrays = (
                np.asarray([r["feature"] for r in s["krr_train"]]),
                np.asarray(s["kernel_alpha"]),
                np.asarray(s["agop_diag"]),
            )
        train, alpha, metric = self._predictor_arrays
        diff = train - np.asarray(feature)
        distance = np.sqrt(np.maximum(np.sum(metric * diff * diff, axis=1), 0.0))
        return float(alpha @ np.exp(-self.parameters["kernel_gamma"] * distance))

    def score(self, feature):
        eta = self.parameters["rfm_influence"] * min(
            1.0,
            self.state["metrics"]["krr_updates"]
            / self.parameters["rfm_influence_warmup_updates"],
        )
        return self.base_score(feature) + eta * self.residual(feature)

    def retrieve(self, query):
        if self._query is not None:
            raise RuntimeError("Previous TokenWorth task has not received an outcome")
        self._query = query
        self._query_embedding = self.encode(query)
        memories = self.state["memories"]
        indices = shortlist(
            query,
            [m["search_text"] for m in memories],
            self.parameters["candidate_count"],
        )
        scored = []
        for index in indices:
            record = memories[index]
            feature = self.feature(self._query_embedding, record)
            scored.append((self.score(feature), record, feature))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        selected = scored[: self.parameters["top_k"]]
        self._selected = {
            m["id"]: {"feature": list(f), "base_score": self.base_score(f)}
            for _, m, f in selected
        }
        self._injected = set()  # The renderer must confirm actual injection.
        return [(score, deepcopy(record)) for score, record, _ in selected]

    def discard_retrieval(self):
        self._query = None
        self._selected = {}
        self._injected = set()

    def confirm_injection(self, ids):
        ids = set(ids)
        if not ids.issubset(self._selected):
            raise ValueError("Feedback cannot credit a non-selected memory")
        self._injected = ids

    def feedback(self, success, snippet="", metadata=None):
        if self._query is None or not isinstance(success, bool):
            raise ValueError(
                "TokenWorth feedback requires one valid external boolean outcome"
            )
        s = self.state
        s["step"] += 1
        by_id = {r["id"]: r for r in s["memories"]}
        for memory_id in sorted(self._injected):
            cached = self._selected[memory_id]
            record = by_id[memory_id]
            self.append_feedback(
                cached["feature"],
                1.0 if success else -1.0,
                memory_id,
                cached["base_score"],
            )
            record["access_count"] += 1
            record["last_access_step"] = s["step"]
            record["help_count" if success else "harm_count"] += 1
            record["utility"] = (
                min(5.0, record["utility"] + 0.25)
                if success
                else max(-2.0, record["utility"] - 0.45)
            )
        if self._injected:
            s["metrics"]["retrieve_count"] += 1
        if success and snippet.strip():
            record = {
                "id": s["next_id"],
                "query": self._query,
                "snippet": snippet,
                "search_text": self._query + "\n" + snippet,
                "embedding": list(self._query_embedding),
                "created_step": s["step"],
                "last_access_step": s["step"],
                "access_count": 0,
                "help_count": 0,
                "harm_count": 0,
                "utility": 1.0,
                "metadata": deepcopy(metadata or {}),
            }
            self.admit(record)
        if self._injected and self.parameters["recent_probe_count"]:
            s["recent_queries"].append(list(self._query_embedding))
            s["recent_queries"] = s["recent_queries"][
                -self.parameters["recent_probe_count"] :
            ]
        self._query = None
        self._selected = {}
        self._injected = set()

    def append_feedback(self, feature, label, memory_id, base_score):
        s, p = self.state, self.parameters
        residual = label - math.tanh(base_score / 2.0)
        s["base_pairs"].append({"feature": self.base_features(feature), "label": label})
        s["base_pairs"] = s["base_pairs"][-p["base_buffer_size"] :]
        pair = {
            "feature": list(feature),
            "label": residual,
            "memory_id": memory_id,
            "step": s["step"],
        }
        s["pairs"].append(pair)
        s["pairs"] = s["pairs"][-p["krr_buffer_size"] :]
        s["agop_pairs"].append(deepcopy(pair))
        s["agop_pairs"] = s["agop_pairs"][-p["agop_buffer_size"] :]
        metrics = s["metrics"]
        metrics["pair_writes"] += 1
        metrics["positive_pair_count" if label > 0 else "negative_pair_count"] += 1
        n = metrics["pair_writes"]
        if n % p["base_update_interval"] == 0:
            self.refit_base()
        agop_due = n % p["agop_update_interval"] == 0
        if agop_due and len(s["agop_pairs"]) >= p["min_pair_count_for_rfm"]:
            self.update_agop()
        if (agop_due or n % p["krr_update_interval"] == 0) and len(s["pairs"]) >= p[
            "min_pair_count_for_rfm"
        ]:
            self.refit_krr()

    def refit_base(self):
        s, p = self.state, self.parameters
        x = np.asarray([r["feature"] for r in s["base_pairs"]])
        y = np.asarray([r["label"] for r in s["base_pairs"]])
        prior = np.asarray(PRIOR)
        weights = np.asarray(s["base_weights"])

        def loss(w):
            return float(
                np.logaddexp(0.0, -y * (x @ w)).sum()
                + p["base_regularization"] * np.sum((w - prior) ** 2)
            )

        for _ in range(100):
            probability = np.exp(-np.logaddexp(0.0, y * (x @ weights)))
            gradient = x.T @ (-y * probability) + 2 * p["base_regularization"] * (
                weights - prior
            )
            if np.max(np.abs(gradient)) < 1e-9:
                break
            curvature = probability * (1 - probability)
            hessian = x.T @ (curvature[:, None] * x) + 2 * p[
                "base_regularization"
            ] * np.eye(4)
            direction = np.linalg.solve(hessian, gradient)
            step, old_loss = 1.0, loss(weights)
            for _ in range(40):
                candidate = weights - step * direction
                if loss(candidate) <= old_loss - 1e-4 * step * float(
                    gradient @ direction
                ) + 1e-14 * max(1.0, abs(old_loss)):
                    weights = candidate
                    break
                step *= 0.5
            else:
                raise RuntimeError(
                    "TokenWorth base optimizer cannot find a descent step"
                )
        else:
            raise RuntimeError("TokenWorth base optimizer did not converge")
        smoothed = (1 - p["base_smoothing"]) * np.asarray(s["base_weights"]) + p[
            "base_smoothing"
        ] * weights
        s["base_weights"] = np.clip(smoothed, 0.0, [2.0, 1.0, 1.0, 1.0]).tolist()
        s["metrics"]["base_updates"] += 1

    def refit_krr(self):
        s, p = self.state, self.parameters
        if len(s["pairs"]) < p["min_pair_count_for_rfm"]:
            return
        train = np.asarray([r["feature"] for r in s["pairs"]])
        labels = np.asarray([r["label"] for r in s["pairs"]])
        diff = train[:, None, :] - train[None, :, :]
        distance = np.sqrt(
            np.maximum(np.sum(np.asarray(s["agop_diag"]) * diff * diff, axis=2), 0.0)
        )
        kernel = np.exp(-p["kernel_gamma"] * distance) + p["ridge_lambda"] * np.eye(
            len(train)
        )
        alpha = np.linalg.solve(kernel, labels)
        if not np.isfinite(alpha).all():
            raise RuntimeError("Non-finite TokenWorth KRR fit")
        s["krr_train"] = deepcopy(s["pairs"])
        s["kernel_alpha"] = alpha.tolist()
        s["metrics"]["krr_updates"] += 1

    def update_agop(self):
        s, p = self.state, self.parameters
        if not s["kernel_alpha"]:
            self.refit_krr()
        train = np.asarray([r["feature"] for r in s["krr_train"]])
        alpha, metric = np.asarray(s["kernel_alpha"]), np.asarray(s["agop_diag"])
        gradients = []
        for pair in s["agop_pairs"]:
            diff = np.asarray(pair["feature"])[None, :] - train
            distance = np.sqrt(np.maximum(np.sum(metric * diff * diff, axis=1), 0.0))
            coefficient = np.zeros_like(distance)
            nz = distance > 0
            coefficient[nz] = (
                -p["kernel_gamma"]
                * alpha[nz]
                * np.exp(-p["kernel_gamma"] * distance[nz])
                / distance[nz]
            )
            gradients.append(np.sum(coefficient[:, None] * metric * diff, axis=0))
        diag = np.mean(np.asarray(gradients) ** 2, axis=0)
        updated = np.clip(
            (1 - p["agop_smoothing"]) * metric + p["agop_smoothing"] * diag,
            p["agop_min_weight"],
            p["agop_max_weight"],
        )
        updated /= np.mean(updated)
        s["agop_diag"] = updated.tolist()
        s["metrics"]["agop_updates"] += 1

    def probe_scores(self, memory, probes):
        return [
            self.score(self.feature(q, memory, step=self.state["step"])) for q in probes
        ]

    def admit(self, memory):
        s, p = self.state, self.parameters
        bank = s["memories"]
        append = len(bank) < p["capacity"]
        probes = (
            [list(self._query_embedding)]
            + s["recent_queries"][-p["recent_probe_count"] :]
            if p["recent_probe_count"]
            else [list(self._query_embedding)]
        )
        if not append:
            similarities = [
                (cosine(memory["embedding"], m["embedding"]), m) for m in bank
            ]
            similarity, nearest = max(
                similarities, key=lambda item: (item[0], -item[1]["id"])
            )
            values = self.probe_scores(memory, probes)
            value = sum(values) / len(values)
            old_values = [
                (sum(self.probe_scores(m, probes)) / len(probes), m) for m in bank
            ]
            best_value = max(v for v, _ in old_values)
            append = 1 - similarity > p["novelty_threshold"] or (
                value > best_value + p["write_gain_margin"]
                and sum(v > 0 for v in values) / len(values)
                >= p["positive_ratio_threshold"]
            )
            if (
                not append
                and nearest["utility"] < 0.5
                and value
                > sum(self.probe_scores(nearest, probes)) / len(probes)
                + p["replacement_value_margin"]
            ):
                bank[bank.index(nearest)] = memory
                s["next_id"] += 1
                s["metrics"]["replace_count"] += 1
                return
        if not append:
            s["metrics"]["write_skip_count"] += 1
            return
        bank.append(memory)
        s["next_id"] += 1
        s["metrics"]["write_count"] += 1
        if len(bank) > p["capacity"]:
            weakest = min(
                bank,
                key=lambda m: (
                    sum(self.probe_scores(m, probes)) / len(probes),
                    m["id"],
                ),
            )
            bank.remove(weakest)
            s["metrics"]["evict_count"] += 1
