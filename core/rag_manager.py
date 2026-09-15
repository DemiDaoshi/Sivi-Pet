import importlib.util
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Модели e5 ожидают префиксы "passage: " для документов и "query: " для запросов.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

REQUIRED_PACKAGES = ("chromadb", "sentence_transformers")

# Порог релевантности (L2-дистанция): ниже — фрагмент считаем подходящим.
# Откалибровано на multilingual-e5-small: свои темы 0.26–0.32, посторонние 0.36–0.47.
MAX_RELEVANT_DISTANCE = 0.34


def missing_dependencies() -> list[str]:
    """Пакеты RAG, которых нет в окружении (проверка без импорта)."""
    return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]


# Ошибка предзагрузки torch: если она есть, RAG в этом процессе не заработает.
_preload_error = None


def preload_dependencies() -> bool:
    """Загружает torch заранее.

    На Windows torch и Qt конфликтуют по DLL: если Qt загрузится первым,
    импорт torch падает с WinError 1114 (c10.dll). Поэтому GUI вызывает это
    до импорта PyQt5. Стоит ~1 с, остальные зависимости остаются ленивыми.
    """
    global _preload_error
    if missing_dependencies():
        return False
    try:
        import torch  # noqa: F401
    except Exception as e:
        _preload_error = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
        print(f"[RAG] Не удалось предзагрузить torch: {type(e).__name__}: {e}")
        return False
    return True


class RagManager:
    """Поиск по data/*.md через эмбеддинги и ChromaDB.

    Без chromadb/sentence-transformers available == False, search() вернёт []
    (приложение продолжает работать как обычный чат).
    """

    def __init__(self, data_dir=None, collection_name="sivi_knowledge",
                 embedding_model=EMBEDDING_MODEL, chunk_size=500, chunk_overlap=50):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.missing_dependencies = missing_dependencies()
        self.error = None

        self._encoder = None
        self._client = None
        self._collection = None
        self._ready = False
        self._indexed_chunks = 0

    def unavailable_reason(self) -> str | None:
        """Почему RAG недоступен, или None если всё в порядке."""
        if self.missing_dependencies:
            return "не установлены пакеты: " + ", ".join(self.missing_dependencies)
        if _preload_error:
            return f"torch не загрузился — {_preload_error}"
        return None

    @property
    def available(self) -> bool:
        """Можно ли пользоваться RAG."""
        return self.unavailable_reason() is None

    def _ensure_ready(self) -> bool:
        """Ленивая инициализация при первом обращении."""
        if self._ready:
            return True
        if not self.available:
            self.error = "Не установлены зависимости: " + ", ".join(self.missing_dependencies)
            return False

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name)
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            self._index_documents()
            self._ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[RAG] Ошибка инициализации: {self.error}")
            return False
        return True

    def _index_documents(self):
        """Читает data_dir, режет на фрагменты и кладёт в ChromaDB."""
        chunks, metadatas, ids = [], [], []
        chunk_id = 0

        for root, _, files in os.walk(self.data_dir):
            for filename in sorted(files):
                if not filename.lower().endswith(".md"):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                except OSError as e:
                    print(f"[RAG] Не удалось прочитать {filename}: {e}")
                    continue

                for chunk in self._split_text(text):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    metadatas.append({"source": filename})
                    ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1

        if not chunks:
            print(f"[RAG] В папке {self.data_dir} не найдено ни одного .md файла")
            return

        # В коллекцию идёт чистый текст, префикс — только для эмбеддинга.
        embeddings = self._encoder.encode(
            [PASSAGE_PREFIX + c for c in chunks], show_progress_bar=True
        ).tolist()

        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self._indexed_chunks = len(chunks)
        print(f"[RAG] Проиндексировано {len(chunks)} фрагментов из {self.data_dir}")

    def _split_text(self, text):
        """Разбиение текста на перекрывающиеся фрагменты."""
        size = self.chunk_size
        overlap = self.chunk_overlap
        if size <= 0:
            raise ValueError("chunk_size должен быть больше нуля")
        if overlap >= size:
            # Иначе шаг станет неположительным и цикл зациклится.
            overlap = size // 4

        step = size - overlap
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += step
        return chunks

    def search_with_scores(self, query, top_k=3):
        """Список (фрагмент, distance); пустой — если RAG недоступен."""
        query = (query or "").strip()
        if not query:
            return []
        if not self._ensure_ready():
            return []

        total = self._collection.count()
        if total == 0:
            return []

        query_embedding = self._encoder.encode([QUERY_PREFIX + query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, total),
            include=["documents", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return list(zip(documents, distances))

    def search(self, query, top_k=3):
        """Релевантные фрагменты; пустой список — если RAG недоступен."""
        return [document for document, _ in self.search_with_scores(query, top_k)]
