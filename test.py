from core.lm_client import LmClient

client = LmClient()
messages = [{"role": "user", "content": "Скажи привет"}]
answer = client.send_message(messages)
print(f"Ответ: {answer}")