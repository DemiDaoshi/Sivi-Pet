"""Генератор черновиков golden set по заметкам data/.

    python eval/make_drafts.py                       все заметки, по 2 вопроса на файл
    python eval/make_drafts.py --per-file 3
    python eval/make_drafts.py --negatives 2
    python eval/make_drafts.py --file "MCP-протокол.md"
    python eval/make_drafts.py --model qwen/qwen3.5-9b

Модель придумывает вопросы и эталоны по тексту заметки, но не проверяет их.
Результат складывается в eval/drafts_<время>.json со пометкой draft. Просмотри
его, поправь формулировки и перенеси подходящие записи в eval/golden_set.json.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

from core.lm_client import LmClient, LmClientError
from core.rag_manager import DEFAULT_DATA_DIR

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PER_FILE = 2
DEFAULT_NEGATIVES = 0
DRAFT_TEMPERATURE = 0.4
DRAFT_MAX_TOKENS = 2048

QUESTION_PROMPT = (
    "Ниже заметка из базы знаний. Составь по ней {count} вопросов, ответы на которые "
    "полностью есть в тексте заметки. Для каждого вопроса дай короткий эталонный ответ "
    "(1-3 предложения) и список ключевых фактов.\n"
    "Верни только JSON-массив без текста вокруг:\n"
    '[{{"question": "...", "reference_answer": "...", "key_facts": ["...", "..."]}}]\n\n'
    "Заметка ({filename}):\n{text}"
)

NEGATIVE_PROMPT = (
    "Ниже заметка из базы знаний. Придумай {count} вопросов на близкую тему, ответа "
    "на которые в заметке НЕТ и модель должна честно сказать, что данных нет.\n"
    "Для каждого вопроса дай короткий эталонный ответ, который описывает правильное "
    "поведение (сообщить об отсутствии данных).\n"
    "Верни только JSON-массив без текста вокруг:\n"
    '[{{"question": "...", "reference_answer": "...", "key_facts": []}}]\n\n'
    "Заметка ({filename}):\n{text}"
)

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def extract_list(text):
    """JSON-массив из ответа модели. Пустой список, если разобрать не удалось."""
    if not text:
        return []
    cleaned = _CODE_FENCE.sub("", text).strip()
    for candidate in (cleaned, _array_match(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return value
    return []


def _array_match(text):
    match = _JSON_ARRAY.search(text or "")
    return match.group(0) if match else None


def note_files(data_dir, only=None):
    if only:
        path = os.path.join(data_dir, only)
        return [path] if os.path.isfile(path) else []
    return [os.path.join(data_dir, name)
            for name in sorted(os.listdir(data_dir))
            if name.lower().endswith(".md")]


def read_note(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def ask_model(client, prompt, prefix, kind, source, filename, items):
    try:
        raw = client.send_message(
            [{"role": "user", "content": prompt}],
            temperature=DRAFT_TEMPERATURE,
            max_tokens=DRAFT_MAX_TOKENS,
        )
    except LmClientError as error:
        print(f"  ! {filename}: {error}")
        return 0

    added = 0
    for index, candidate in enumerate(extract_list(raw), start=1):
        question = str(candidate.get("question") or "").strip()
        if not question:
            continue
        key_facts = candidate.get("key_facts") or []
        items.append({
            "id": f"{prefix}-{kind}-{index:02d}",
            "type": "answerable" if kind == "qa" else "unanswerable",
            "draft": True,
            "question": question,
            "expected_sources": [source] if kind == "qa" else [],
            "reference_answer": str(candidate.get("reference_answer") or "").strip(),
            "key_facts": [str(fact).strip() for fact in key_facts if str(fact).strip()],
            "note": "Сгенерировано автоматически, не вычитано.",
        })
        added += 1
    return added


def slug_for(filename):
    stem = os.path.splitext(filename)[0].lower()
    stem = re.sub(r"\s+", "-", stem.strip())
    return re.sub(r"[^0-9a-zа-яё\-]+", "", stem)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Черновики golden set по заметкам")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--file", help="только одна заметка, по имени")
    parser.add_argument("--per-file", type=int, default=DEFAULT_PER_FILE)
    parser.add_argument("--negatives", type=int, default=DEFAULT_NEGATIVES)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = note_files(args.data_dir, args.file)
    if not files:
        print(f"В {args.data_dir} не нашлось .md файлов. Положи заметки и повтори.")
        return 1

    client = LmClient(model=args.model)
    items = []
    for path in files:
        filename = os.path.basename(path)
        text = read_note(path)
        print(f"[{filename}]")
        prefix = slug_for(filename)

        if args.per_file > 0:
            prompt = QUESTION_PROMPT.format(count=args.per_file, filename=filename, text=text)
            added = ask_model(client, prompt, prefix, "qa", filename, filename, items)
            print(f"  = {added} записей с ответом из заметки")

        if args.negatives > 0:
            prompt = NEGATIVE_PROMPT.format(count=args.negatives, filename=filename, text=text)
            added = ask_model(client, prompt, prefix, "neg", filename, filename, items)
            print(f"  = {added} записей без ответа")

    out_path = args.out or os.path.join(
        EVAL_DIR, f"drafts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Черновики, сгенерированные моделью. Требуют вычитки перед переносом в golden_set.json.",
        "items": items,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\nГотово: {len(items)} черновиков -> {out_path}")
    print("Проверь их и перенеси лучшие в eval/golden_set.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
