import importlib.util
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Модели e5 ожидают префиксы "passage: " для документов и "query: " для запросов.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

REQUIRED_PACKAGES = ("chromadb", "sentence_transformers")


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

        self.missing_dependencies = [name for name in REQUIRED_PACKAGES
                                     if importlib.util.find_spec(name) is None]
        self.error = None

        self._encoder = None
        self._client = None
        self._collection = None
        self._ready = False
        self._indexed_chunks = 0

    @property
    def available(self) -> bool:
        """Установлены ли зависимости RAG."""
        return not self.missing_dependencies

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

    def search(self, query, top_k=3):
        """Релевантные фрагменты; пустой список — если RAG недоступен."""
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
        )
        documents = results.get("documents") or []
        return list(documents[0]) if documents else []
