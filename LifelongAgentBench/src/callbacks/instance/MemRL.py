import os

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import SampleStatus, SessionEvaluationOutcome
from .advanced_memory_utils import (
    MemRLEpisodeRecord,
    build_memrl_episode_record,
    select_memrl_candidate_indices,
)
from .lifelong_memory_utils import (
    dump_json_state_list,
    inject_memory_block,
    load_json_state_list,
    truncate_text,
    truncate_text_list_total,
)


class MemRLCallback(Callback):
    def __init__(
        self,
        top_k: int,
        candidate_count: int = 16,
        learning_rate: float = 0.3,
        max_snippet_length: int = 1600,
        minimum_relevance_score: int = 2,
        store_incorrect_episodes: bool = False,
        max_memory_count: int = 256,
        max_injected_chars_total: int = 1600,
        use_evaluator_outcome_only: bool = False,
    ):
        super().__init__()
        assert top_k > 0
        assert candidate_count >= top_k
        assert 0.0 < learning_rate <= 1.0
        assert minimum_relevance_score > 0
        assert max_memory_count > 0
        assert max_injected_chars_total > 0
        self.top_k = top_k
        self.candidate_count = candidate_count
        self.learning_rate = learning_rate
        self.max_snippet_length = max_snippet_length
        self.minimum_relevance_score = minimum_relevance_score
        self.store_incorrect_episodes = store_incorrect_episodes
        self.max_memory_count = max_memory_count
        self.max_injected_chars_total = max_injected_chars_total
        self.use_evaluator_outcome_only = use_evaluator_outcome_only
        self.episode_record_list: list[MemRLEpisodeRecord] = []
        self.current_step = 0
        self.current_retrieved_record_index_list: list[int] = []

    def _get_memory_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memrl_episode_record_list.json")

    def _get_step_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memrl_step.txt")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def restore_state(self) -> None:
        episode_record_list = [
            MemRLEpisodeRecord.model_validate(item)
            for item in load_json_state_list(
                self._get_memory_state_path(), MemRLEpisodeRecord
            )
        ]
        self.episode_record_list = [
            record
            for record in episode_record_list
            if self.store_incorrect_episodes
            or record.outcome == SessionEvaluationOutcome.CORRECT
        ]
        if os.path.exists(self._get_step_state_path()):
            self.current_step = int(
                open(self._get_step_state_path(), "r").read().strip()
            )

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        if (
            len(self.episode_record_list) == 0
            or session.sample_status != SampleStatus.RUNNING
            or session.chat_history.get_value_length() == 0
        ):
            self.current_retrieved_record_index_list = []
            return
        current_query = session.chat_history.get_item_deep_copy(-1).content
        selected_index_list = select_memrl_candidate_indices(
            current_query,
            self.episode_record_list,
            self.top_k,
            self.candidate_count,
            self.current_step,
            self.minimum_relevance_score,
        )
        for index in selected_index_list:
            record = self.episode_record_list[index]
            record.last_access_step = self.current_step
            record.retrieve_count += 1
        self.current_retrieved_record_index_list = selected_index_list
        inject_memory_block(
            session.chat_history,
            "MemRL episodic memories",
            truncate_text_list_total(
                [self.episode_record_list[index].snippet for index in selected_index_list],
                self.max_injected_chars_total,
            ),
        )

    def _update_q_value(self, record: MemRLEpisodeRecord, reward: float) -> None:
        record.q_value = record.q_value + self.learning_rate * (reward - record.q_value)
        if reward > 0:
            record.success_count += 1
        else:
            record.failure_count += 1

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        self.current_step += 1
        session = callback_args.current_session
        is_correct = (
            session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            and (
                self.use_evaluator_outcome_only
                or session.sample_status == SampleStatus.COMPLETED
            )
        )
        reward = (
            1.0
            if is_correct
            else -1.0
        )
        for index in self.current_retrieved_record_index_list:
            self._update_q_value(self.episode_record_list[index], reward)
            self.episode_record_list[index].last_access_step = self.current_step
        should_process_session = (
            self.use_evaluator_outcome_only
            or session.sample_status == SampleStatus.COMPLETED
        )
        if not should_process_session or (
            not is_correct and not self.store_incorrect_episodes
        ):
            return
        episode_record = build_memrl_episode_record(
            session,
            callback_args.session_context.agent.get_role_dict(),
            self.max_snippet_length,
            0.0,
            self.current_step,
        )
        episode_record.snippet = truncate_text(
            episode_record.snippet,
            self.max_snippet_length,
        )
        self.episode_record_list.append(episode_record)
        if len(self.episode_record_list) > self.max_memory_count:
            self.episode_record_list.sort(
                key=lambda record: (
                    record.q_value,
                    record.retrieve_count,
                    record.last_access_step,
                ),
                reverse=True,
            )
            self.episode_record_list = self.episode_record_list[: self.max_memory_count]

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_memory_state_path(), self.episode_record_list)
        open(self._get_step_state_path(), "w").write(str(self.current_step))
