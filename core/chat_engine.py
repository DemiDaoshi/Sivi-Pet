from typing import NamedTuple

from core.lm_client import LmClient
from core.tool_manager import ToolManager


class ChatResult(NamedTuple):
    answer: str
    used_rag: bool


class ChatEngine:
    """Один ход диалога с обработкой RAG. Общий для консоли и GUI."""

    def __init__(self, client: LmClient, history, tools: ToolManager):
        self.client = client
        self.history = history
        self.tools = tools

    def send(self, user_text: str) -> ChatResult:
        self.history.add_message("user", user_text)

        answer = self.client.send_message(self.history.messages)

        # Проверяем, не запросила ли модель инструмент.
        tool_result = self.tools.handle(answer)
        if tool_result is None:
            self.history.add_message("assistant", answer)
            return ChatResult(answer, used_rag=False)

        # Контекст уходит ролью user: system в середине диалога даёт 400.
        request_messages = self.history.messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": tool_result},
        ]
        answer = self.tools.strip_marker(self.client.send_message(request_messages))
        self.history.add_message("assistant", answer)
        return ChatResult(answer, used_rag=True)
