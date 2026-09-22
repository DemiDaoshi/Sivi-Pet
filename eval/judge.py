"""LLM-судья для оценки ответов Sivi.

Модель в проекте одна, поэтому судья — это та же модель, но с отдельным
контекстом: на каждый вопрос собирается свежий список сообщений, без истории
диалога и без системного промпта Sivi. Судья видит только вопрос, ответ,
найденный контекст и (не всегда) эталон, и возвращает разбор в JSON.
"""
import json
import re
from typing import NamedTuple

from core.lm_client import LmClient

JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024

JUDGE_SYSTEM_PROMPT = (
    "Ты — строгий судья качества ответов RAG-ассистента. Тебе дают вопрос, "
    "ответ ассистента, найденный в базе знаний контекст и необязательный эталонный ответ.\n"
    "Оценивай только то, что видишь. Не решай задачу заново и не добавляй знания извне.\n"
    "Поля разбора:\n"
    "- relevant: ответ отвечает именно на заданный вопрос.\n"
    "- grounded: каждый факт ответа подтверждается найденным контекстом. "
    "Если контекста нет, верни null. Если контекст есть, а факта в нём нет — false.\n"
    "- correct: ответ совпадает с эталоном по смыслу. Если эталон не задан, верни null.\n"
    "- score: целое от 1 до 5, общая оценка ответа.\n"
    "- reason: одна короткая фраза с объяснением.\n"
    "Верни только JSON без текста вокруг: "
    '{"score": 5, "relevant": true, "grounded": true, "correct": true, "reason": "..."}'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def extract_json(text):
    """Первый JSON-объект из ответа модели. None, если разобрать не удалось."""
    if not text:
        return None

    cleaned = _CODE_FENCE.sub("", text).strip()
    for candidate in (cleaned, _first_match(_JSON_OBJECT, text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _first_match(pattern, text):
    match = pattern.search(text or "")
    return match.group(0) if match else None


class JudgeVerdict(NamedTuple):
    score: int | None = None
    correct: bool | None = None
    grounded: bool | None = None
    relevant: bool | None = None
    reason: str = ""
    raw: str = ""


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "да", "yes", "1"}:
        return True
    if text in {"false", "нет", "no", "0"}:
        return False
    return None


def _as_score(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def parse_verdict(raw):
    """Разбор ответа судьи. Терпим к лишнему тексту вокруг JSON."""
    data = extract_json(raw) or {}
    reason = str(data.get("reason") or "").strip()
    raw = raw or ""
    return JudgeVerdict(
        score=_as_score(data.get("score")),
        correct=_as_bool(data.get("correct")),
        grounded=_as_bool(data.get("grounded")),
        relevant=_as_bool(data.get("relevant")),
        reason=reason or raw.strip()[:300],
        raw=raw,
    )


def format_contexts(contexts):
    """Найденные фрагменты как текст для судьи."""
    if not contexts:
        return "(контекст не найден)"
    blocks = []
    for index, chunk in enumerate(contexts, start=1):
        text = getattr(chunk, "text", chunk)
        source = getattr(chunk, "source", "") or "без источника"
        distance = getattr(chunk, "distance", None)
        suffix = f" (distance {distance:.3f})" if isinstance(distance, (int, float)) else ""
        blocks.append(f"[{index}] {source}{suffix}\n{text}")
    return "\n\n".join(blocks)


def verdict_line(verdict):
    """Короткая строка вердикта для консоли, GUI и отчёта."""
    if verdict is None:
        return "нет вердикта"
    parts = []
    if verdict.score is not None:
        parts.append(f"оценка {verdict.score}/5")
    for label, value in (("верно", verdict.correct),
                         ("обосновано", verdict.grounded),
                         ("по теме", verdict.relevant)):
        if value is not None:
            parts.append(label if value else f"не {label}")
    return " · ".join(parts) if parts else "нет разбора"


class Judge:
    def __init__(self, client: LmClient, temperature=JUDGE_TEMPERATURE,
                 max_tokens=JUDGE_MAX_TOKENS):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_messages(self, question, answer, contexts=(), reference=None):
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._task(question, answer, contexts, reference)},
        ]

    def _task(self, question, answer, contexts, reference):
        return (
            f"Вопрос пользователя:\n{question}\n\n"
            f"Эталонный ответ:\n{reference or 'не задан'}\n\n"
            f"Найденный контекст:\n{format_contexts(contexts)}\n\n"
            f"Ответ ассистента:\n{answer}"
        )

    def evaluate(self, question, answer, contexts=(), reference=None) -> JudgeVerdict:
        messages = self.build_messages(question, answer, contexts, reference)
        raw = self.client.send_message(
            messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        return parse_verdict(raw)
