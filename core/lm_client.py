# core/lm_client.py
import requests

class LmClient:
    def __init__(self, base_url="http://localhost:1234/v1/chat/completions"):
        self.url = base_url

    def send_message(self, messages: list, temperature=0.7, max_tokens=100) -> str:
        """Отправляет историю сообщений модели и возвращает ответную строку."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ошибка связи с LM Studio: {e}")
