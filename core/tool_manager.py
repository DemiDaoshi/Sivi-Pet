import re
from typing import NamedTuple

from core.rag_manager import MAX_RELEVANT_DISTANCE, RetrievedChunk

# Модель отвечает "[RAG: запрос]"; разбор терпим к регистру и пробелам.
RAG_PATTERN = re.compile(r"\[\s*RAG\s*:\s*(?P<query>.+?)\s*\]", re.IGNORECASE | re.DOTALL)


class ToolResult(NamedTuple):
    text: str
    chunks: tuple[RetrievedChunk, ...] = ()


class ToolManager:
    def __init__(self, rag_manager, top_k=3, max_distance=MAX_RELEVANT_DISTANCE):
        self.rag = rag_manager
        self.top_k = top_k
        self.max_distance = max_distance

    def extract_query(self, answer: str) -> str | None:
        """Запрос из маркера [RAG: ...] или None."""
        if not answer:
            return None
        match = RAG_PATTERN.search(answer)
        if not match:
            return None
        query = match.group("query").strip()
        return query or None

    def strip_marker(self, text: str) -> str:
        """Убирает служебные маркеры [RAG: ...] из ответа модели."""
        return RAG_PATTERN.sub("", text or "").strip()

    def context_for(self, query: str) -> tuple[str | None, list[RetrievedChunk]]:
        """Ищет по базе принудительно. Возвращает (текст контекста, найденные фрагменты)."""
        if not self.rag.available:
            return None, []

        found = [chunk for chunk in self.rag.search_with_meta(query, top_k=self.top_k)
                 if chunk.distance is not None and chunk.distance <= self.max_distance]
        if not found:
            return None, []

        formatted = "\n\n".join(
            f"--- Фрагмент {i + 1} ---\n{chunk.text}" for i, chunk in enumerate(found)
        )
        return (f"Контекст из базы знаний по запросу «{query}»:\n\n{formatted}\n\n"
                "Ответь пользователю, опираясь на этот контекст."), found

    def handle(self, answer: str) -> ToolResult | None:
        """Инструмент, запрошенный моделью. Возвращает ToolResult или None."""
        query = self.extract_query(answer)
        if query is None:
            return None

        if not self.rag.available:
            return ToolResult(
                f"Контекст из базы знаний недоступен ({self.rag.unavailable_reason()}). "
                "Ответь пользователю, опираясь только на свои знания."
            )

        context, chunks = self.context_for(query)
        if not chunks:
            return ToolResult(
                f"В базе знаний ничего не найдено по запросу «{query}». "
                "Ответь пользователю, опираясь только на свои знания."
            )
        return ToolResult(context, tuple(chunks))
