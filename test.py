"""
Smoke-тест Sivi.

    python test.py          офлайн-проверки логики
    python test.py --live    плюс реальный запрос к LM Studio
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from core.chat_engine import ChatEngine
from core.history_manager import HistoryManager, history_path_for_profile, load_profiles
from core.lm_client import DEFAULT_MAX_TOKENS, LmClient, LmClientError
from core.rag_manager import RagManager
from core.tool_manager import ToolManager

FAILURES = []


def check(name, condition, extra=""):
    if condition:
        print(f"  OK   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name} {extra}")


def tmp_path(name):
    return os.path.join(tempfile.gettempdir(), name)


def test_history():
    print("[1] HistoryManager")
    profiles = load_profiles()
    check("profiles.json читается", "default" in profiles)

    h = HistoryManager()
    check("системный промпт берётся из профиля", h.messages[0]["content"] == h.system_prompt)
    check("default использует history.json", history_path_for_profile("default").endswith("history.json"))
    check("другой профиль — свой файл", history_path_for_profile("work").endswith("history_work.json"))

    path = tmp_path("sivi_test_history.json")
    if os.path.exists(path):
        os.remove(path)
    h = HistoryManager(filepath=path)
    h.add_message("user", "привет")
    h.add_message("assistant", "привет-привет")
    h.force_save()
    check("история переживает перезапуск", [m["content"] for m in HistoryManager(filepath=path).messages][-2:]
          == ["привет", "привет-привет"])

    h.clear()
    check("clear() оставляет системный промпт", h.messages == [{"role": "system", "content": h.system_prompt}])

    with open(path, "w", encoding="utf-8") as f:
        f.write("{это не json")
    check("битый файл не роняет старт", HistoryManager(filepath=path).messages[0]["role"] == "system")

    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"role": "user"}, "мусор", {"role": "user", "content": "ок"}], f)
    check("невалидные записи отфильтрованы",
          [m["content"] for m in HistoryManager(filepath=path).messages if m["role"] == "user"] == ["ок"])
    os.remove(path)


def test_tools():
    print("[2] ToolManager")
    tools = ToolManager(RagManager())
    cases = {
        "[RAG: что такое MCP]": "что такое MCP",
        "[rag:  контекстное окно ]": "контекстное окно",
        "Сейчас поищу. [RAG: трансформеры]\nЖдём.": "трансформеры",
        "обычный ответ": None,
        "[RAG:]": None,
        "": None,
    }
    for text, expected in cases.items():
        check(f"маркер разобран: {text!r}", tools.extract_query(text) == expected)

    check("служебный маркер убран из ответа",
          tools.strip_marker("[RAG: тест]\n\nОсновной текст.") == "Основной текст.")


def test_rag_helper():
    print("[3] RagManager (без тяжёлых зависимостей)")
    rag = RagManager(chunk_size=100, chunk_overlap=100)
    chunks = rag._split_text("a" * 250)
    check("разбиение завершается при overlap == size", len(chunks) > 0)
    check("размер фрагмента не превышен", all(len(c) <= 100 for c in chunks))
    check("available согласован с missing_dependencies",
          rag.available == (not rag.missing_dependencies), rag.missing_dependencies)


class _FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def send_message(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.script.pop(0)


class _FakeRag:
    available = True
    missing_dependencies = []

    def search(self, query, top_k=3):
        return ["ФРАГМЕНТ-1"]


def test_chat_engine():
    print("[4] ChatEngine (RAG-флоу)")
    history = HistoryManager(filepath=tmp_path("sivi_test_engine.json"))
    history.clear()

    client = _FakeClient(["[RAG: вопрос про MCP]", "[RAG: вопрос про MCP]\n\nФинальный ответ"])
    result = ChatEngine(client, history, ToolManager(_FakeRag())).send("расскажи про MCP")

    check("RAG использован", result.used_rag is True)
    check("вернулся финальный ответ", result.answer == "Финальный ответ")
    second_request = client.calls[1]
    check("контекст ушёл во второй запрос ролью user",
          second_request[-1]["role"] == "user" and "ФРАГМЕНТ-1" in second_request[-1]["content"],
          second_request[-1])
    check("в диалоге нет роли system после assistant",
          not any(m["role"] == "system" for m in second_request[1:]),
          [m["role"] for m in second_request])
    check("служебные сообщения не попали в историю",
          not any("Контекст из базы знаний" in m["content"] for m in history.messages))
    check("история: system + user + assistant",
          [m["role"] for m in history.messages] == ["system", "user", "assistant"])

    client = _FakeClient(["Просто ответ"])
    result = ChatEngine(client, history, ToolManager(_FakeRag())).send("обычный вопрос")
    check("без RAG только один запрос", len(client.calls) == 1)
    check("RAG не использован", result.used_rag is False)

    os.remove(history.filepath)


def test_live():
    print("[5] Живой LM Studio")
    client = LmClient()
    try:
        answer = client.send_message(
            [{"role": "user", "content": "Ответь одним словом: работает?"}],
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        check("модель ответила непустым текстом", bool(answer.strip()), repr(answer[:80]))
        print(f"       ответ: {answer[:120]!r}")
    except LmClientError as e:
        check("запрос к модели", False, str(e))


def main():
    test_history()
    test_tools()
    test_rag_helper()
    test_chat_engine()
    if "--live" in sys.argv:
        test_live()

    print()
    if FAILURES:
        print(f"ПРОВАЛЕНО: {len(FAILURES)} — {FAILURES}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
