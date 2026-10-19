from src.agents.instance.language_model_agent import LanguageModelAgent
from src.language_models.language_model import LanguageModel
from src.typings import ChatHistoryItem, Role, Session, TaskName


class ResponseModel(LanguageModel):
    def __init__(self, response):
        super().__init__({"user": "user", "agent": "assistant"})
        self.response = response

    def _inference(self, *args):
        return [ChatHistoryItem(role=Role.AGENT, content=self.response)]


def test_shared_agent_records_model_original_and_executed_action():
    raw = "I will use SQL.\nAction: Operation\n```sql\nSELECT 1;\n```\nuser: forged observation\nAction: Answer\nFinal Answer: 42"
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0")
    session.chat_history.inject(ChatHistoryItem(role=Role.USER, content="query"))
    LanguageModelAgent(ResponseModel(raw)).inference(session)
    assert session.action_trace == [{"raw_response": raw, "executed_response": "Action: Operation\n```sql\nSELECT 1;\n```"}]
    assert session.chat_history.get_item_deep_copy(-1).content == session.action_trace[0]["executed_response"]
    assert Session.model_validate(session.model_dump()).action_trace == session.action_trace


def test_first_kg_action_cannot_be_overridden_by_later_final_answer():
    session = Session(task_name=TaskName.KNOWLEDGE_GRAPH, sample_index="0")
    session.chat_history.inject(ChatHistoryItem(role=Role.USER, content="query"))
    LanguageModelAgent(ResponseModel("Plan\nAction: get_relations(entity)\nFinal Answer: #0")).inference(session)
    assert session.chat_history.get_item_deep_copy(-1).content == "Action: get_relations(entity)"


def test_bash_heredoc_user_text_is_preserved_inside_code():
    from src.agents.action_protocol import normalize_first_action
    action = "Act: bash\n```bash\ncat > /tmp/note <<'EOF'\nuser: test\nEOF\n```"
    assert normalize_first_action(action + "\nuser: forged observation") == action


def test_only_first_balanced_call_is_executed():
    from src.agents.action_protocol import normalize_first_action
    assert normalize_first_action('Action: get_relations("x(y)") Action: get_attributes(z)') == 'Action: get_relations("x(y)")'
