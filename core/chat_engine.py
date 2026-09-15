from typing import NamedTuple

from core.lm_client import LmClient
from core.tool_manager import ToolManager


class ChatResult(NamedTuple):
    answer: str
    used_rag: bool
    fragments: int = 0


class ChatEngine:
    """Один ход диалога с обработкой RAG. Общий для консоли и GUI."""

    def __init__(self, client: LmClient, history, tools: ToolManager):
        self.client = client
        self.history = history
        self.tools = tools

    def send(self, user_text: str, force_rag: bool = False) -> ChatResult:
        """force_rag — искать по базе принудительно, не спрашивая модель."""
        self.history.add_message("user", user_text)

        if force_rag:
            context, found = self.tools.context_for(user_text)
            if found:
                return self._answer_with_context(context, found)

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

    def _answer_with_context(self, context: str, found: int) -> ChatResult:
        request_messages = self.history.messages + [{"role": "user", "content": context}]
        answer = self.tools.strip_marker(self.client.send_message(request_messages))
        self.history.add_message("assistant", answer)
        return ChatResult(answer, used_rag=True, fragments=found)
