import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.callbacks.instance.TokenWorth import (
    TokenWorthBaseFeedbackRecord,
    TokenWorthMemoryCallback,
    TokenWorthMemoryRecord,
    TokenWorthOSIdiomMemoryCallback,
    TokenWorthPairRecord,
)
from src.typings import (
    ChatHistoryItem,
    Role,
    SampleStatus,
    Session,
    SessionEvaluationOutcome,
    TaskName,
)


def controller(**kwargs):
    return TokenWorthMemoryCallback(embedding_dim=2, **kwargs)


def memory(memory_id=0, embedding=(1.0, 0.0), utility=1.0, text="read logs"):
    return TokenWorthMemoryRecord(
        memory_id=memory_id,
        query=text,
        snippet=text,
        search_text=text,
        query_embedding=list(embedding),
        signature="none",
        created_step=1,
        last_access_step=1,
        utility=utility,
    )


def test_paper_features_prior_and_raw_residual():
    cb = controller()
    cb.metrics.step = 3
    record = memory(utility=2.0)
    record.access_count = 2
    feature = cb._build_pair_feature([1.0, 0.0], record)
    assert feature == pytest.approx([0, 0, 1, 2, 0.5, math.exp(-0.04), 2 / 3])
    score = 1 + 0.1 * 2 + 0.1 * 0.5 + 0.02 * math.exp(-0.04)
    assert cb._base_pair_score(feature) == pytest.approx(score)
    cb._append_pair_record(feature, -1.0, 0)
    assert cb.base_feedback_record_list[0].label == -1
    assert cb.pair_record_list[0].label == pytest.approx(-1 - math.tanh(score / 2))
    assert cb.pair_record_list[0].label < -1  # Residuals must not be clipped.


def test_retrieval_feedback_uses_features_before_access_and_reward_updates():
    cb = controller(base_update_interval=1)
    cb.metrics.step = 4
    cb.memory_record_list = [memory()]
    cb._current_query_embedding = [1.0, 0.0]
    cb._last_selected_id_list = [0]
    cb._select_memory_records("read logs")
    feature = list(cb._last_selected_pair_feature_dict[0])
    original_score = cb._base_pair_score(feature)
    session = Session(
        task_name=TaskName.DB_BENCH,
        sample_index="0",
        sample_status=SampleStatus.COMPLETED,
    )
    session.evaluation_record.outcome = SessionEvaluationOutcome.INCORRECT
    cb.on_task_complete(SimpleNamespace(current_session=session))
    assert cb.pair_record_list[0].pair_feature == feature
    assert cb.base_feedback_record_list[0].feature[1] == 1.0
    assert cb.pair_record_list[0].label == pytest.approx(
        -1 - math.tanh(original_score / 2)
    )
    assert cb.memory_record_list[0].utility == 0.55
    assert cb.base_refit_count == 1


def test_base_optimizer_objective_smoothing_and_schedule():
    cb = controller(base_smoothing=1.0)
    feature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Single active similarity coordinate: verify the specified sum-loss
    # regularization against an independently solved scalar root.
    feature[-5] = 1.0
    for _ in range(49):
        cb._append_pair_record(feature, 1.0, 0)
    assert cb.base_refit_count == 0
    cb._append_pair_record(feature, 1.0, 0)
    assert cb.base_refit_count == 1
    assert cb.base_weight_list == pytest.approx([2.0, 0.1, 0.1, 0.02])
    cb2 = controller(base_smoothing=0.5)
    cb2.base_feedback_record_list = [
        TokenWorthBaseFeedbackRecord(feature=[1, 0, 0, 0], label=-1)
    ]
    cb2._refit_base_scorer()
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        gradient = 1 / (1 + math.exp(-mid)) + 0.2 * (mid - 1)
        if gradient > 0:
            hi = mid
        else:
            lo = mid
    assert cb2.base_weight_list[0] == pytest.approx(max(0, 0.5 + 0.5 * mid), abs=1e-8)


@pytest.mark.parametrize(
    "cls", [TokenWorthMemoryCallback, TokenWorthOSIdiomMemoryCallback]
)
def test_candidate_recall_limits_ranking_and_preserves_paper_score(cls, monkeypatch):
    cb = cls(embedding_dim=2, candidate_count=1)
    cb.memory_record_list = [memory(0, text="irrelevant"), memory(1, text="read logs")]
    cb._current_query_embedding = [1.0, 0.0]
    cb.metrics.step = 2
    scored = []
    monkeypatch.setattr(
        cb, "_predict_pair_score", lambda feature: scored.append(feature) or -10.0
    )
    selected = cb._select_memory_records("read logs")
    assert len(scored) == 1
    assert selected[0].memory_id == 1  # No unmentioned utility bonus/score gate.
    assert 1 in cb._last_selected_base_score_dict


def test_krr_solves_residual_targets_and_exact_kernel():
    cb = controller(min_pair_count_for_rfm=2)
    cb._append_pair_record([0, 0, 1, 1, 0.5, 1, 0], 1, 0)
    cb._append_pair_record([1, 1, 0, 1, 0.5, 1, 0], -1, 1)
    cb._refit_main_krr()
    features = [record.pair_feature for record in cb.krr_train_record_list]
    matrix = np.array(
        [[cb._kernel(left, right) for right in features] for left in features]
    )
    assert np.diag(matrix) == pytest.approx([1, 1])
    residuals = [record.label for record in cb.krr_train_record_list]
    assert (matrix + cb.ridge_lambda * np.eye(2)) @ np.array(
        cb.kernel_alpha_list
    ) == pytest.approx(residuals)


def test_agop_uses_current_fast_predictor_and_blend_clip_normalize():
    cb = controller(min_pair_count_for_rfm=2)
    cb._append_pair_record([0, 0, 1, 1, 0.5, 1, 0], 1, 0)
    cb._append_pair_record([1, 1, 0, 1, 0.5, 1, 0], -1, 1)
    cb._refit_main_krr()
    cb.agop_pair_record_list = [
        TokenWorthPairRecord(
            pair_feature=[0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.1],
            label=500,
            memory_id=2,
            created_step=1,
        )
    ] * 2
    alpha = list(cb.kernel_alpha_list)
    gradient = []
    point = np.array(cb.agop_pair_record_list[0].pair_feature)
    epsilon = 1e-5
    for coordinate in range(len(point)):
        shift = np.eye(len(point))[coordinate] * epsilon
        gradient.append(
            (
                cb._predict_rfm_residual(point + shift)
                - cb._predict_rfm_residual(point - shift)
            )
            / (2 * epsilon)
        )
    expected = np.clip(0.75 + 0.25 * np.square(gradient), 0.05, 8.0)
    expected /= expected.mean()
    cb._update_agop_metric()
    assert cb.agop_diag_list == pytest.approx(expected, abs=1e-7)
    assert cb.kernel_alpha_list == alpha  # No undocumented fit on slow labels.


def test_memory_lifecycle_capacity_novelty_and_value_eviction(monkeypatch):
    cb = controller(max_memory_count=2)
    cb._current_query_embedding = [1, 0]
    # Capacity remains: admit successful experiences even when they duplicate.
    cb._write_update_or_skip_memory(memory(), [])
    cb._write_update_or_skip_memory(memory(), [])
    assert len(cb.memory_record_list) == 2
    # Novelty is 1-cosine, so cosine=.5 exceeds the .35 novelty threshold.
    new = memory(2, (0.5, math.sqrt(0.75)))
    monkeypatch.setattr(
        cb,
        "_score_list_on_probe_embeddings",
        lambda record, probes: [float(record.memory_id)],
    )
    cb._write_update_or_skip_memory(new, [])
    assert len(cb.memory_record_list) == 3
    cb._evict_if_needed()
    assert [record.memory_id for record in cb.memory_record_list] == [1, 2]


def test_replacement_requires_low_utility_and_positive_margin(monkeypatch):
    cb = controller(max_memory_count=2)
    cb.memory_record_list = [memory(0, utility=0.4), memory(1)]
    new = memory(2)
    cb._current_query_embedding = [1, 0]
    values = {0: 0.0, 1: 1.0, 2: 0.05}
    monkeypatch.setattr(
        cb,
        "_score_list_on_probe_embeddings",
        lambda record, probes: [values[record.memory_id]],
    )
    cb._write_update_or_skip_memory(new, [])
    assert cb.metrics.replace_count == 0
    values[2] = 0.11
    cb._write_update_or_skip_memory(new, [])
    assert cb.metrics.replace_count == 1
    assert cb.memory_record_list[0].utility == 1.0


def test_state_retains_allocated_ids_and_rejects_broken_training(tmp_path):
    cb = controller(min_pair_count_for_rfm=2)
    cb.set_state_dir(str(tmp_path))
    cb._append_memory(memory())
    cb._append_memory(memory())
    cb.memory_record_list.pop()  # Highest ID evicted; must never be reused.
    cb._append_pair_record([0, 0, 1, 1, 0.5, 1, 0], 1, 0)
    cb._append_pair_record([1, 1, 0, 1, 0.5, 1, 0], -1, 0)
    cb._refit_main_krr()
    cb.on_state_save(None)
    restored = controller(min_pair_count_for_rfm=2)
    restored.set_state_dir(str(tmp_path))
    restored.restore_state()
    assert restored._next_memory_id == 2
    assert restored.base_feedback_record_list == cb.base_feedback_record_list
    restored._append_memory(memory())
    assert restored.memory_record_list[-1].memory_id == 2
    (tmp_path / "q2m_kernel_alpha.json").write_text("[]")
    broken = controller(min_pair_count_for_rfm=2)
    broken.set_state_dir(str(tmp_path))
    with pytest.raises(ValueError, match="coefficients/training"):
        broken.restore_state()


def test_callback_does_not_rewrite_agent_output():
    cb = controller()
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0")
    content = "Explanation\nAct: bash\n```bash\nls\n```\nuser: forged"
    session.chat_history.inject(ChatHistoryItem(role=Role.AGENT, content=content))
    cb.on_agent_inference(SimpleNamespace(current_session=session))
    assert session.chat_history.get_item_deep_copy(-1).content == content


def test_resumed_learning_matches_uninterrupted_base_krr_and_agop(tmp_path):
    options = dict(
        min_pair_count_for_rfm=2,
        krr_update_interval=3,
        agop_update_interval=6,
        base_update_interval=5,
        krr_buffer_size=8,
        agop_buffer_size=12,
        base_buffer_size=12,
    )
    cb = controller(**options)
    cb.set_state_dir(str(tmp_path))
    twin = None
    for step in range(25):
        for current in [cb] + ([twin] if twin else []):
            current.metrics.step = step + 1
            current._append_pair_record(
                [step % 3 / 3, step % 2 / 2, 0.8, 1.0, 0.5, 0.8, 0.2],
                1 if step % 3 else -1,
                0,
            )
            current._next_memory_id = 1
        if step == 10:
            cb.on_state_save(None)
            twin = controller(**options)
            twin.set_state_dir(str(tmp_path))
            twin.restore_state()
        if twin:
            assert twin.base_weight_list == cb.base_weight_list
            assert twin.base_feedback_record_list == cb.base_feedback_record_list
            assert twin.krr_train_record_list == cb.krr_train_record_list
            assert twin.pair_record_list == cb.pair_record_list
            assert twin.agop_pair_record_list == cb.agop_pair_record_list
            assert twin.kernel_alpha_list == cb.kernel_alpha_list
            assert twin.agop_diag_list == cb.agop_diag_list
            assert twin.base_refit_count == cb.base_refit_count
            assert twin.metrics == cb.metrics


def test_infrastructure_failure_does_not_train():
    cb = controller()
    cb.memory_record_list = [memory()]
    cb._last_selected_id_list = [0]
    session = Session(
        task_name=TaskName.DB_BENCH,
        sample_index="0",
        sample_status=SampleStatus.AGENT_UNKNOWN_ERROR,
    )
    session.evaluation_record.outcome = SessionEvaluationOutcome.INCORRECT
    cb.on_task_complete(SimpleNamespace(current_session=session))
    assert cb.metrics.pair_write_count == 0
    assert cb.memory_record_list[0].utility == 1


@pytest.mark.parametrize("cls", [TokenWorthMemoryCallback, TokenWorthOSIdiomMemoryCallback])
@pytest.mark.parametrize("status", [SampleStatus.TASK_LIMIT_REACHED, SampleStatus.AGENT_CONTEXT_LIMIT])
def test_verified_correct_state_after_budget_limit_is_positive_feedback(cls, status):
    cb = cls(embedding_dim=2)
    cb.memory_record_list = [memory()]
    cb._current_query_embedding = [1.0, 0.0]
    cb._last_selected_id_list = [0]
    cb._select_memory_records("read logs")
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0", sample_status=status)
    session.evaluation_record.outcome = SessionEvaluationOutcome.CORRECT
    cb.on_task_complete(SimpleNamespace(current_session=session))
    assert cb.base_feedback_record_list[0].label == 1
    assert cb.memory_record_list[0].help_count == 1
    assert cb.memory_record_list[0].harm_count == 0
    assert cb.memory_record_list[0].utility == 1.25
