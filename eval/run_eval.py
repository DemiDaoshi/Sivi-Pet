"""Прогон golden set: метрики поиска и оценка ответов судьёй.

    python eval/run_eval.py                    полный прогон
    python eval/run_eval.py --no-judge         только метрики поиска
    python eval/run_eval.py --limit 3          первые три вопроса
    python eval/run_eval.py --types answerable,unanswerable
    python eval/run_eval.py --model qwen/qwen3.5-9b

На каждый вопрос: сначала честный поиск по базе (метрики), затем один ход
ChatEngine в свежей истории, затем вердикт судьи с отдельным контекстом.
Отчёт складывается в eval/results/ в виде Markdown и JSON.
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

import requests

from core.chat_engine import ChatEngine
from core.history_manager import HistoryManager
from core.lm_client import LmClient, LmClientError
from core.rag_manager import MAX_RELEVANT_DISTANCE, RagManager
from core.tool_manager import ToolManager
from eval.judge import Judge, JudgeVerdict, verdict_line
from eval.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank, share_true

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

GOLDEN_TYPES = ("answerable", "open", "unanswerable")
DEFAULT_TOP_K = 3


class Palette:
    """ANSI-цвета для консольного прогона. Отключается на не-tty и при NO_COLOR."""

    def __init__(self, enabled):
        self.enabled = enabled

    def paint(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def head(self, text):
        return self.paint(text, "1")

    def model(self, text):
        return self.paint(text, "36")

    def dim(self, text):
        return self.paint(text, "90")

    def good(self, text):
        return self.paint(text, "32")

    def mid(self, text):
        return self.paint(text, "33")

    def bad(self, text):
        return self.paint(text, "31")


def supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def load_golden_set(path=GOLDEN_SET_PATH):
    """Читает и проверяет golden set. Падает с понятным текстом, если что-то не так."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("ожидается список вопросов или объект с полем items")

    problems = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"запись {index}: не объект")
            continue
        label = item.get("id") or f"запись {index}"
        for field in ("id", "question", "type"):
            if not item.get(field):
                problems.append(f"{label}: нет поля {field}")
        if item.get("type") not in GOLDEN_TYPES:
            problems.append(f"{label}: неизвестный type={item.get('type')!r}")
        if item.get("type") == "answerable" and not item.get("expected_sources"):
            problems.append(f"{label}: для answerable нужен непустой expected_sources")

    if problems:
        raise ValueError("проблемы в golden set:\n  " + "\n  ".join(problems))
    return items


def one_line(text, limit=400):
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def retrieval_row(item, rag, top_k):
    retrieved = rag.search_with_meta(item["question"], top_k=top_k)
    sources = [chunk.source for chunk in retrieved]
    distances = [chunk.distance for chunk in retrieved if chunk.distance is not None]
    expected = item.get("expected_sources") or []
    relevant = [chunk for chunk in retrieved
                if chunk.distance is not None and chunk.distance <= MAX_RELEVANT_DISTANCE]
    return {
        "retrieved": [{"source": chunk.source, "distance": chunk.distance} for chunk in retrieved],
        "relevant_count": len(relevant),
        "top_distance": min(distances) if distances else None,
        "metrics": {
            f"hit@{top_k}": hit_at_k(sources, expected, top_k),
            f"recall@{top_k}": recall_at_k(sources, expected, top_k),
            "mrr": reciprocal_rank(sources, expected),
            f"precision@{top_k}": precision_at_k(sources, expected, top_k),
        },
    }


def run_item(item, rag, client, history, judge, top_k):
    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "expected_sources": item.get("expected_sources") or [],
        "relevant_count": 0,
        "top_distance": None,
        "retrieved": [],
        "metrics": {},
        "answer": None,
        "used_rag": None,
        "fragments": None,
        "sources": [],
        "judge": None,
        "error": None,
    }
    row.update(retrieval_row(item, rag, top_k))

    history.clear()
    tools = ToolManager(rag, top_k=top_k)
    engine = ChatEngine(client, history, tools)
    try:
        result = engine.send(item["question"], force_rag=True)
    except LmClientError as error:
        row["error"] = str(error)
        return row

    row["answer"] = result.answer
    row["used_rag"] = result.used_rag
    row["fragments"] = result.fragments
    row["sources"] = [{"source": chunk.source, "distance": chunk.distance}
                      for chunk in result.sources]

    if judge is not None:
        try:
            row["judge"] = judge.evaluate(
                item["question"], result.answer, result.sources,
                reference=item.get("reference_answer"),
            )
        except LmClientError as error:
            row["error"] = f"судья: {error}"
    return row


def summarize(rows, top_k):
    answerable = [row for row in rows if row["type"] == "answerable"]
    without_expected = [row for row in rows if not row["expected_sources"]]
    judged = [row for row in rows if row["judge"] is not None]
    judged_answerable = [row for row in answerable if row["judge"] is not None]

    def metric(name, subset):
        return mean([row["metrics"].get(name) for row in subset])

    return {
        "total": len(rows),
        "by_type": {kind: sum(1 for row in rows if row["type"] == kind) for kind in GOLDEN_TYPES},
        "errors": sum(1 for row in rows if row["error"]),
        "retrieval": {
            f"hit@{top_k}": metric(f"hit@{top_k}", answerable),
            f"recall@{top_k}": metric(f"recall@{top_k}", answerable),
            "mrr": metric("mrr", answerable),
            f"precision@{top_k}": metric(f"precision@{top_k}", answerable),
            "avg_top1_answerable": mean([row["top_distance"] for row in answerable]),
            "avg_top1_without_expected": mean([row["top_distance"] for row in without_expected]),
            "empty_search_share": share_true([row["relevant_count"] == 0 for row in without_expected]),
        },
        "answers": {
            "judged": len(judged),
            "avg_score": mean([row["judge"].score for row in judged]),
            "relevant_share": share_true([row["judge"].relevant for row in judged]),
            "grounded_share": share_true([row["judge"].grounded for row in judged_answerable]),
            "correct_share": share_true([row["judge"].correct for row in judged]),
        },
    }


def failed_rows(rows):
    failures = []
    for row in rows:
        if row["error"]:
            failures.append((row["id"], row["error"]))
            continue
        verdict = row["judge"]
        if verdict is None:
            continue
        bad_score = verdict.score is not None and verdict.score <= 2
        if bad_score or verdict.correct is False or verdict.grounded is False:
            failures.append((row["id"], verdict_line(verdict)))
    return failures


def verdict_color(palette, verdict):
    if verdict is None:
        return palette.dim
    if verdict.score is not None and verdict.score >= 4:
        return palette.good
    if verdict.score is not None and verdict.score <= 2:
        return palette.bad
    return palette.mid


def print_row(row, palette, top_k):
    print()
    print(palette.head("-" * 60))
    print(palette.head(f"{row['id']}  [{row['type']}]"))
    print(f"Вопрос: {row['question']}")

    if row["retrieved"]:
        found = ", ".join(f"{chunk['source']} ({chunk['distance']:.3f})"
                          for chunk in row["retrieved"])
    else:
        found = "ничего"
    print(palette.dim(f"Поиск: {found} | прошло порог: {row['relevant_count']}"))

    if row["error"]:
        print(palette.bad(f"Ошибка: {row['error']}"))
        return

    print(palette.model(f"МОДЕЛЬ: {one_line(row['answer'])}"))
    verdict = row["judge"]
    if verdict is not None:
        color = verdict_color(palette, verdict)
        print(color(f"СУДЬЯ: {verdict_line(verdict)}"))
        if verdict.reason:
            print(color(f"       причина: {one_line(verdict.reason, 300)}"))


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_summary(summary, palette, top_k, max_distance):
    print()
    print(palette.head("=" * 60))
    print(palette.head(f"Итог: вопросов {summary['total']}, ошибок {summary['errors']}"))
    print(palette.dim(f"top_k={top_k}, порог={max_distance}"))
    print("Метрики поиска (только answerable):")
    for name in (f"hit@{top_k}", f"recall@{top_k}", "mrr", f"precision@{top_k}"):
        print(f"  {name}: {fmt(summary['retrieval'][name])}")
    print("Поведение без ответа в базе:")
    print(f"  средний top-1 distance: {fmt(summary['retrieval']['avg_top1_without_expected'])}")
    print(f"  доля пустой выдачи: {fmt(summary['retrieval']['empty_search_share'])}")
    answers = summary["answers"]
    if answers["judged"]:
        print(f"Судья (оценено {answers['judged']}):")
        print(f"  средняя оценка: {fmt(answers['avg_score'], 2)}")
        print(f"  по теме: {fmt(answers['relevant_share'])}")
        print(f"  обосновано контекстом: {fmt(answers['grounded_share'])}")
        print(f"  верно по эталону: {fmt(answers['correct_share'])}")


def build_markdown(report):
    summary = report["summary"]
    date = report["generated_at"]
    lines = [
        f"# Отчёт eval — {date}",
        "",
        f"- Модель: {report['model'] or 'по умолчанию (LM Studio)'}",
        f"- Вопросов: {summary['total']} "
        f"({', '.join(f'{kind}: {count}' for kind, count in summary['by_type'].items())})",
        f"- top_k: {report['top_k']}, порог: {report['max_distance']}",
        f"- Ошибок: {summary['errors']}",
        "",
        "## Метрики поиска",
        "",
        "| Метрика | Значение |",
        "| --- | --- |",
    ]
    for name, value in summary["retrieval"].items():
        lines.append(f"| {name} | {fmt(value)} |")

    lines += [
        "",
        "## Оценки судьи",
        "",
        "| Метрика | Значение |",
        "| --- | --- |",
    ]
    for name, value in summary["answers"].items():
        lines.append(f"| {name} | {fmt(value, 2)} |")

    if report["failures"]:
        lines += ["", "## Что не прошло", "", "| id | Причина |", "| --- | --- |"]
        for item_id, reason in report["failures"]:
            lines.append(f"| {item_id} | {one_line(reason, 200)} |")

    lines += ["", "## По вопросам", ""]
    for row in report["rows"]:
        lines.append(f"### {row['id']} · {row['type']}")
        lines.append("")
        lines.append(f"**Вопрос:** {row['question']}")
        lines.append("")
        found = ", ".join(f"{chunk['source']} ({chunk['distance']:.3f})"
                          for chunk in row["retrieved"]) or "ничего"
        lines.append(f"**Поиск:** {found} | прошло порог: {row['relevant_count']}")
        lines.append("")
        if row["error"]:
            lines.append(f"**Ошибка:** {row['error']}")
            lines.append("")
            continue
        lines.append("**Ответ модели:**")
        lines.append("")
        lines.append(f"> {one_line(row['answer'], 1500)}")
        lines.append("")
        verdict = row["judge"]
        if isinstance(verdict, JudgeVerdict):
            lines.append(f"**Судья:** {verdict_line(verdict)}")
            lines.append("")
            lines.append(f"**Причина:** {one_line(verdict.reason, 500)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def json_ready(report):
    report = dict(report)
    rows = []
    for row in report["rows"]:
        row = dict(row)
        verdict = row.get("judge")
        if isinstance(verdict, JudgeVerdict):
            row["judge"] = verdict._asdict()
        rows.append(row)
    report["rows"] = rows
    return report


def available_models():
    """Список моделей LM Studio для подсказки в консоли. Пустой, если сервер молчит."""
    url = LmClient().url.rsplit("/chat/completions", 1)[0] + "/models"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return [entry.get("id") for entry in response.json().get("data", []) if entry.get("id")]
    except (requests.exceptions.RequestException, ValueError):
        return []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Прогон golden set Sivi")
    parser.add_argument("--golden", default=GOLDEN_SET_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--types", help="типы через запятую")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-judge", action="store_true", help="только метрики поиска")
    parser.add_argument("--out", default=None, help="путь Markdown-отчёта")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    palette = Palette(supports_color())

    try:
        items = load_golden_set(args.golden)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(palette.bad(f"Не удалось прочитать golden set: {error}"))
        return 2

    if args.types:
        wanted = {name.strip() for name in args.types.split(",") if name.strip()}
        items = [item for item in items if item["type"] in wanted]
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(palette.bad("Нет вопросов для прогона."))
        return 1

    rag = RagManager()
    if not rag.available:
        print(palette.bad(f"RAG недоступен: {rag.unavailable_reason()}"))
        return 2

    client = LmClient(model=args.model, temperature=0.0)
    judge = None if args.no_judge else Judge(client)
    history = HistoryManager(filepath=os.path.join(tempfile.gettempdir(), "sivi_eval_history.json"))

    if args.model is None:
        models = available_models()
        if len(models) > 1:
            print(palette.mid("В LM Studio загружено несколько моделей. Без --model запрос может "
                              f"упасть с ошибкой 400. Доступны: {', '.join(models)}"))
        elif models:
            print(palette.dim(f"Модель не указана, используется {models[0]}"))

    print(palette.head(f"Прогон golden set: {len(items)} вопросов"))
    print(palette.dim(f"top_k={args.top_k}, порог={MAX_RELEVANT_DISTANCE}, "
                      f"судья={'выключен' if judge is None else 'включён'}"))

    rows = []
    for index, item in enumerate(items, start=1):
        print(palette.dim(f"\n[{index}/{len(items)}] {item['id']}"))
        row = run_item(item, rag, client, history, judge, args.top_k)
        rows.append(row)
        print_row(row, palette, args.top_k)

    summary = summarize(rows, args.top_k)
    print_summary(summary, palette, args.top_k, MAX_RELEVANT_DISTANCE)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "top_k": args.top_k,
        "max_distance": MAX_RELEVANT_DISTANCE,
        "summary": summary,
        "failures": failed_rows(rows),
        "rows": rows,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = args.out or os.path.join(RESULTS_DIR, f"report_{stamp}.md")
    json_path = os.path.splitext(md_path)[0] + ".json"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(build_markdown(report))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, ensure_ascii=False, indent=2)

    print()
    print(palette.head(f"Markdown: {md_path}"))
    print(palette.head(f"JSON:     {json_path}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
