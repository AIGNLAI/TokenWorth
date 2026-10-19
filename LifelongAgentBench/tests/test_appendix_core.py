"""Independent scientific checks for the appendix algorithms and LifelongAgentBench parity."""

import copy
import math

import numpy as np
import pytest

from tokenworth_core import TokenWorthCore, encode, shortlist
from src.callbacks.instance.TokenWorth import TokenWorthMemoryCallback


def record(memory_id=0, embedding=(1.0, 0.0), utility=1.0, query="read logs"):
    return dict(
        id=memory_id,
        embedding=list(embedding),
        utility=utility,
        query=query,
        snippet=query,
        search_text=query,
        created_step=1,
        last_access_step=1,
        access_count=0,
        help_count=0,
        harm_count=0,
        metadata={},
    )


def test_hash_encoder_matches_lifelongagentbench():
    cb = TokenWorthMemoryCallback(embedding_dim=512)
    for text in (
        "",
        "read JSON_file 123",
        "中文 and punctuation!?",
        "put an apple in fridge",
    ):
        assert encode(text) == cb._encode_dense_vector(text)


def test_full_feature_order_and_retrieval_time_residual():
    core = TokenWorthCore({"embedding_dim": 2, "base_update_interval": 1})
    core.state["step"] = 3
    core.state["memories"] = [record(utility=2.0)]
    core.state["next_id"] = 1
    core.encode = lambda text: [1.0, 0.0]
    before = copy.deepcopy(core.state)
    core.retrieve("read logs")
    assert core.state == before  # no persisted retrieval-time mutations
    cached = copy.deepcopy(core._selected[0])
    assert cached["feature"] == pytest.approx([0, 0, 1, 2, 0.5, math.exp(-0.06), 0])
    core.confirm_injection([0])
    core.feedback(False)
    pair = core.state["pairs"][0]
    assert pair["feature"] == cached["feature"]
    assert pair["label"] == pytest.approx(-1 - math.tanh(cached["base_score"] / 2))
    assert pair["label"] < -1
    assert core.state["base_pairs"][0]["label"] == -1
    assert core.state["memories"][0]["utility"] == pytest.approx(1.55)
    assert core.state["memories"][0]["access_count"] == 1
    assert core.state["metrics"]["base_updates"] == 1


def test_uninjected_memory_is_not_supervised():
    core = TokenWorthCore({"embedding_dim": 2})
    core.state["memories"] = [record()]
    core.state["next_id"] = 1
    core.retrieve("read logs")
    core.confirm_injection([])
    core.feedback(False)
    assert core.state["pairs"] == []
    assert core.state["memories"][0]["utility"] == 1
    assert core.state["memories"][0]["access_count"] == 0


def test_bm25_recall_precedes_worth_and_has_no_hidden_gate():
    core = TokenWorthCore({"embedding_dim": 2, "candidate_count": 1})
    core.state["memories"] = [
        record(0, utility=5, query="totally different"),
        record(1, utility=-2, query="read logs"),
    ]
    core.state["next_id"] = 2
    core.encode = lambda query: [1.0, 0.0]
    selected = core.retrieve("read logs")
    assert selected[0][1]["id"] == 1  # high-utility lexical mismatch is not recalled
    assert shortlist("read logs", ["different", "read logs"], 1) == [1]


def test_independent_krr_equation_and_no_prediction_clipping():
    core = TokenWorthCore({"embedding_dim": 2, "min_pair_count_for_rfm": 1})
    core.state["pairs"] = [
        dict(feature=[0, 0, 1, 1, 0.5, 1, 0], label=3.0, memory_id=0, step=1)
    ]
    core.refit_krr()
    assert core.state["kernel_alpha"] == pytest.approx([3 / 1.01])
    assert (
        core.kernel(
            core.state["pairs"][0]["feature"], core.state["pairs"][0]["feature"]
        )
        == 1
    )
    assert core.residual(core.state["pairs"][0]["feature"]) > 1


def test_core_matches_lifelongagentbench_learning_clocks_and_optimizer():
    params = dict(
        embedding_dim=2,
        krr_update_interval=3,
        agop_update_interval=9,
        krr_buffer_size=8,
        agop_buffer_size=16,
        base_buffer_size=16,
        base_update_interval=5,
        min_pair_count_for_rfm=2,
    )
    core = TokenWorthCore(params)
    cb = TokenWorthMemoryCallback(**params)
    cb._ensure_agop_diag(7)
    for i in range(1, 81):
        feature = [
            i % 3 / 10.0,
            i % 5 / 10.0,
            i % 7 / 10.0,
            i % 4 - 0.5,
            0.5,
            math.exp(-0.02 * (i % 5)),
            i % 2 / 10.0,
        ]
        y = 1.0 if i % 3 else -1.0
        score = core.base_score(feature)
        core.state["step"] = i
        cb.metrics.step = i
        core.append_feedback(feature, y, 0, score)
        cb._last_selected_base_score_dict[0] = score
        cb._append_pair_record(feature, y, 0)
        assert core.state["base_weights"] == pytest.approx(
            cb.base_weight_list, abs=1e-10
        )
        assert core.state["agop_diag"] == pytest.approx(cb.agop_diag_list, abs=1e-10)
        assert core.state["kernel_alpha"] == pytest.approx(
            cb.kernel_alpha_list, abs=1e-9
        )
        assert core.state["metrics"]["krr_updates"] == cb.hybrid_krr_refit_count
        assert core.state["metrics"]["agop_updates"] == cb.hybrid_agop_update_count
        assert core.state["pairs"][-1]["label"] == pytest.approx(
            cb.pair_record_list[-1].label
        )
    assert core.state["metrics"]["agop_updates"] > 0


def test_value_only_eviction_and_weak_neighbor_replacement():
    core = TokenWorthCore({"embedding_dim": 2, "capacity": 1})
    core.state["memories"] = [record(utility=0.1)]
    core.state["next_id"] = 1
    core._query_embedding = [1.0, 0.0]
    # No novelty, no gain over the BEST existing memory. Replacement only
    # happens through the explicit weak-neighbor branch when gain exceeds .1.
    core.parameters["write_gain_margin"] = 10.0
    core.probe_scores = lambda memory, probes: [memory["utility"]] * len(probes)
    core.admit(record(1, utility=1.0))
    assert core.state["metrics"]["replace_count"] == 1
    assert core.state["memories"][0]["id"] == 1
    core.admit(record(2, embedding=(0, 1), utility=-1.0))
    assert len(core.state["memories"]) == 1
    assert core.state["memories"][0]["id"] == 1
    assert core.state["metrics"]["evict_count"] == 1


def test_resumed_long_stream_is_identical_and_bounded():
    parameters = dict(
        embedding_dim=8,
        capacity=4,
        krr_buffer_size=8,
        agop_buffer_size=16,
        base_buffer_size=16,
        krr_update_interval=3,
        agop_update_interval=9,
        base_update_interval=5,
        min_pair_count_for_rfm=2,
    )
    core = TokenWorthCore(parameters)
    twin = None
    for i in range(230):
        for method in (core, twin):
            if method is None:
                continue
            selected = method.retrieve("read logs {}".format(i % 9))
            method.confirm_injection([r["id"] for _, r in selected])
            method.feedback(i % 3 != 0, "verified action {}".format(i))
            method.validate_state()
        if i == 111:
            twin = TokenWorthCore(parameters, copy.deepcopy(core.state))
        if twin is not None:
            assert twin.state == core.state
    assert len(core.state["memories"]) <= 4
    assert core.state["metrics"]["base_updates"] > 0
    assert core.state["metrics"]["agop_updates"] > 0


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.pop("base_pairs"),
        lambda s: s.update(version="legacy"),
        lambda s: s["kernel_alpha"].append(1),
        lambda s: s["agop_diag"].__setitem__(0, float("nan")),
    ],
)
def test_incomplete_or_legacy_state_is_rejected(change):
    state = copy.deepcopy(TokenWorthCore().state)
    change(state)
    with pytest.raises(ValueError):
        TokenWorthCore(state=state)


def test_released_dimensions_and_learning_intervals_survive_full_buffers():
    core = TokenWorthCore({"capacity": 50})
    for i in range(270):
        selected = core.retrieve("Read structured logs for operation {}".format(i % 13))
        core.confirm_injection([r["id"] for _, r in selected])
        core.feedback(i % 4 != 0, "Verified parser action {}".format(i))
    core.validate_state()
    assert len(core.state["base_pairs"]) == 256
    assert len(core.state["pairs"]) == 128
    assert len(core.state["agop_pairs"]) == 256
    assert core.state["metrics"]["base_updates"] == 5
    assert core.state["metrics"]["agop_updates"] == 1
    assert core.state["metrics"]["krr_updates"] == 3
    restored = TokenWorthCore({"capacity": 50}, copy.deepcopy(core.state))
    assert restored.state == core.state
