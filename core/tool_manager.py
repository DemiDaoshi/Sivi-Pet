import re

# Модель отвечает "[RAG: запрос]"; разбор терпим к регистру и пробелам.
RAG_PATTERN = re.compile(r"\[\s*RAG\s*:\s*(?P<query>.+?)\s*\]", re.IGNORECASE | re.DOTALL)


class ToolManager:
    def __init__(self, rag_manager, top_k=3):
        self.rag = rag_manager
        self.top_k = top_k

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

    def handle(self, answer: str) -> str | None:
        """Выполняет инструмент из ответа модели. Возвращает контекст или None."""
        query = self.extract_query(answer)
        if query is None:
            return None

        if not self.rag.available:
            return ("Контекст из базы знаний недоступен (не установлены зависимости: "
                    f"{', '.join(self.rag.missing_dependencies)}). "
                    "Ответь пользователю, опираясь только на свои знания.")

        chunks = self.rag.search(query, top_k=self.top_k)
        if not chunks:
            return (f"В базе знаний ничего не найдено по запросу «{query}». "
                    "Ответь пользователю, опираясь только на свои знания.")

        formatted = "\n\n".join(
            f"--- Фрагмент {i + 1} ---\n{chunk}" for i, chunk in enumerate(chunks)
        )
        return (f"Контекст из базы знаний по запросу «{query}»:\n\n{formatted}\n\n"
                "Используй этот контекст, чтобы ответить пользователю.")
