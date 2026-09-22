"""Метрики поиска для golden set.

Чистые функции: на вход имена найденных файлов и ожидаемые источники, на выход
числа. Ни модели, ни ChromaDB здесь нет, поэтому эти функции легко проверить
в test.py. Сравнение идёт по именам файлов (source), а не по тексту фрагментов:
это устойчивее к нарезке на чанки и понятнее в отчёте.
"""


def _top_k(values, k):
    values = list(values or [])
    return values if k is None else values[:k]


def hit_at_k(retrieved_sources, expected_sources, k=None):
    """Есть ли хотя бы один ожидаемый источник в первых k. None, если ожидаемых нет."""
    expected = set(expected_sources or [])
    if not expected:
        return None
    return bool(expected & set(_top_k(retrieved_sources, k)))


def recall_at_k(retrieved_sources, expected_sources, k=None):
    """Какая доля ожидаемых источников попала в первые k. None, если ожидаемых нет."""
    expected = set(expected_sources or [])
    if not expected:
        return None
    found = expected & set(_top_k(retrieved_sources, k))
    return len(found) / len(expected)


def reciprocal_rank(retrieved_sources, expected_sources):
    """1 / позиция первого попадания. 0.0, если попаданий нет."""
    expected = set(expected_sources or [])
    if not expected:
        return None
    for rank, source in enumerate(retrieved_sources or [], start=1):
        if source in expected:
            return 1 / rank
    return 0.0


def precision_at_k(retrieved_sources, expected_sources, k=None):
    """Какая доля выдачи оказалась ожидаемыми источниками. None, если ожидаемых нет."""
    expected = set(expected_sources or [])
    if not expected:
        return None
    top = _top_k(retrieved_sources, k)
    if not top:
        return 0.0
    return sum(1 for source in top if source in expected) / len(top)


def mean(values):
    """Среднее без учёта None. None, если считать нечего."""
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def share_true(values):
    """Доля значений True без учёта None. None, если считать нечего."""
    flags = [value for value in values if value is not None]
    if not flags:
        return None
    return sum(1 for flag in flags if flag) / len(flags)
