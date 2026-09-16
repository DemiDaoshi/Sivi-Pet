import requests

# Локальный OpenAI-совместимый сервер LM Studio.
DEFAULT_BASE_URL = "http://localhost:1234/v1/chat/completions"

# reasoning-моделям нужен запас токенов и времени (особенно с RAG-контекстом).
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 300


class LmClientError(Exception):
    """Ошибка обращения к LLM: сеть, HTTP, некорректный или пустой ответ."""


class LmClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, model=None,
                 temperature=0.65, max_tokens=DEFAULT_MAX_TOKENS, timeout=DEFAULT_TIMEOUT):
        self.url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def send_message(self, messages: list, temperature=None, max_tokens=None, model=None) -> str:
        """Отправляет историю сообщений модели и возвращает текст ответа."""
        payload = {
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        chosen_model = model if model is not None else self.model
        if chosen_model:
            payload["model"] = chosen_model

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise LmClientError(f"Ошибка связи с LM Studio ({self.url}): {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            raise LmClientError(f"Сервер вернул не JSON: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            raise LmClientError(f"В ответе нет choices: {str(data)[:300]}")

        choice = choices[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()

        if content:
            return content

        # Пустой content — типичная ситуация для reasoning-моделей.
        reasoning = message.get("reasoning_content") or ""
        if choice.get("finish_reason") == "length":
            raise LmClientError(
                "Модель израсходовала лимит max_tokens на reasoning и не успела ответить. "
                f"Увеличьте max_tokens (сейчас {payload['max_tokens']})."
            )
        if reasoning:
            raise LmClientError("Модель вернула только reasoning_content без ответа. Попробуйте ещё раз.")
        raise LmClientError("Модель вернула пустой ответ.")
