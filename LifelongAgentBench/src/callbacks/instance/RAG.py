import os

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import SessionEvaluationOutcome, SampleStatus
from .lifelong_memory_utils import (
    ExperienceRecord,
    build_session_snippet,
    dump_json_state_list,
    inject_memory_block,
    load_json_state_list,
    rank_texts_by_relevance,
    truncate_text,
    truncate_text_list_total,
)


class RAGExperienceRetrievalCallback(Callback):
    def __init__(
        self,
        top_k: int,
        max_snippet_length: int = 1600,
        max_memory_count: int = 256,
        max_injected_chars_total: int = 1600,
        use_evaluator_outcome_only: bool = False,
    ):
        super().__init__()
        assert top_k > 0
        assert max_snippet_length > 0
        assert max_memory_count > 0
        assert max_injected_chars_total > 0
        self.top_k = top_k
        self.max_snippet_length = max_snippet_length
        self.max_memory_count = max_memory_count
        self.max_injected_chars_total = max_injected_chars_total
        self.use_evaluator_outcome_only = use_evaluator_outcome_only
        self.experience_record_list: list[ExperienceRecord] = []

    def _get_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "experience_record_list.json")

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def restore_state(self) -> None:
        self.experience_record_list = [
            ExperienceRecord.model_validate(item)
            for item in load_json_state_list(self._get_state_path(), ExperienceRecord)
        ]

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        # 如果经验池是空的，直接返回
        if len(self.experience_record_list) == 0:
            return

        chat_history = callback_args.current_session.chat_history

        # 检查对话历史是否为空
        if chat_history.get_value_length() == 0:
            # 如果历史为空，尝试从任务工厂构造初始指令作为查询词 (Query)
            # 这里的 0 代表获取第 0 条消息，即初始指令
            try:
                current_query = callback_args.session_context.task.system_prompt_factory.construct(
                    0, expected_role=callback_args.session_context.agent.get_role_dict()
                ).content
            except Exception:
                # 如果还是拿不到，说明任务还没初始化好，直接跳过本次 RAG
                return
        else:
            # 如果历史不为空，取最后一条内容作为查询词
            current_query = chat_history.get_item_deep_copy(-1).content

        # 如果查询词还是空的，也没法做 RAG
        if not current_query:
            return

        # 执行 BM25 检索
        selected_index_list = rank_texts_by_relevance(
            current_query,
            [record.search_text for record in self.experience_record_list],
            self.top_k,
        )

        # 注入检索到的经验
        inject_memory_block(
            chat_history,
            "Retrieved relevant past experiences",
            truncate_text_list_total(
                [
                    self.experience_record_list[index].snippet
                    for index in selected_index_list
                ],
                self.max_injected_chars_total,
            ),
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        current_session = callback_args.current_session
        is_correct = (
            current_session.evaluation_record.outcome
            == SessionEvaluationOutcome.CORRECT
        )
        if not is_correct or (
            not self.use_evaluator_outcome_only
            and current_session.sample_status != SampleStatus.COMPLETED
        ):
            return
        role_dict = callback_args.session_context.agent.get_role_dict()
        experience_record = build_session_snippet(
            current_session,
            role_dict,
            self.max_snippet_length,
        )
        experience_record.snippet = truncate_text(
            experience_record.snippet,
            self.max_snippet_length,
        )
        self.experience_record_list.append(experience_record)
        if len(self.experience_record_list) > self.max_memory_count:
            self.experience_record_list = self.experience_record_list[
                -self.max_memory_count :
            ]

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        dump_json_state_list(self._get_state_path(), self.experience_record_list)
