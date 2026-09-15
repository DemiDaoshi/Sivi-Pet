from core.chat_engine import ChatEngine
from core.history_manager import HistoryManager
from core.lm_client import LmClient, LmClientError
from core.rag_manager import RagManager
from core.tool_manager import ToolManager

EXIT_COMMANDS = {"выход", "exit", "quit"}
CLEAR_COMMANDS = {"забудь всё", "забудь все", "очисти", "clear"}
RAG_ON_COMMANDS = {"раг вкл", "поиск вкл", "rag on"}
RAG_OFF_COMMANDS = {"раг выкл", "поиск выкл", "rag off"}


def main():
    client = LmClient()
    history = HistoryManager()
    rag = RagManager()
    engine = ChatEngine(client, history, ToolManager(rag))

    if rag.available:
        print("[RAG] База знаний готова (индексация при первом обращении к ней).")
    else:
        print(f"[RAG] Недоступен: нет пакетов {', '.join(rag.missing_dependencies)}. "
              "Работаю как обычный чат.")

    print("Чат запущен. Команды: 'выход', 'забудь всё', 'раг вкл' / 'раг выкл'.")

    force_rag = False
    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in EXIT_COMMANDS:
            break
        if command in CLEAR_COMMANDS:
            history.clear()
            print("История очищена.")
            continue
        if command in RAG_ON_COMMANDS:
            force_rag = True
            print("Поиск по базе знаний включён.")
            continue
        if command in RAG_OFF_COMMANDS:
            force_rag = False
            print("Поиск по базе знаний выключен: решает модель.")
            continue

        try:
            result = engine.send(user_input, force_rag=force_rag)
        except LmClientError as e:
            print(f"Ошибка: {e}")
            history.force_save()
            continue
        except KeyboardInterrupt:
            print("\nПрервано.")
            break

        marker = " [RAG]" if result.used_rag else ""
        print(f"Ассистент{marker}: {result.answer}")
        if force_rag and not result.used_rag:
            print("  (в базе знаний ничего подходящего не найдено)")

    history.force_save()
    print("Пока!")


if __name__ == "__main__":
    main()
