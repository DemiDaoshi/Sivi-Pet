import re

from core.rag_manager import MAX_RELEVANT_DISTANCE

# Модель отвечает "[RAG: запрос]"; разбор терпим к регистру и пробелам.
RAG_PATTERN = re.compile(r"\[\s*RAG\s*:\s*(?P<query>.+?)\s*\]", re.IGNORECASE | re.DOTALL)


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

    def context_for(self, query: str) -> tuple[str | None, int]:
        """Ищет по базе принудительно. Возвращает (текст контекста, число фрагментов)."""
        if not self.rag.available:
            return None, 0

        found = [document for document, distance
                 in self.rag.search_with_scores(query, top_k=self.top_k)
                 if distance <= self.max_distance]
        if not found:
            return None, 0

        formatted = "\n\n".join(
            f"--- Фрагмент {i + 1} ---\n{chunk}" for i, chunk in enumerate(found)
        )
        return (f"Контекст из базы знаний по запросу «{query}»:\n\n{formatted}\n\n"
                "Ответь пользователю, опираясь на этот контекст."), len(found)

    def handle(self, answer: str) -> str | None:
        """Инструмент, запрошенный моделью. Возвращает текст или None."""
        query = self.extract_query(answer)
        if query is None:
            return None

        if not self.rag.available:
            return (f"Контекст из базы знаний недоступен ({self.rag.unavailable_reason()}). "
                    "Ответь пользователю, опираясь только на свои знания.")

        context, found = self.context_for(query)
        if not found:
            return (f"В базе знаний ничего не найдено по запросу «{query}». "
                    "Ответь пользователю, опираясь только на свои знания.")
        return context
