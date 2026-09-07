# main_console.py
from core.lm_client import LmClient
from core.history_manager import HistoryManager

client = LmClient()
history = HistoryManager()

print("Чат запущен. 'выход' для завершения.")
while True:
    user_input = input("Ты: ")
    if user_input.lower() == "выход":
        history.force_save()
        print("Пока!")
        break
    if user_input.lower() == "забудь всё":
        history.clear()
        print("История очищена.")
        continue


    history.add_message("user", user_input)
    try:
        answer = client.send_message(history.messages)
        print(f"Ассистент: {answer}")
        history.add_message("assistant", answer)
    except ConnectionError as e:
        print(f"Ошибка: {e}")