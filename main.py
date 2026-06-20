import os


from openai import OpenAI
import psycopg
from psycopg.types.json import Jsonb


# === Настройки ===

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/ragdb",
)

LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL",
    "http://127.0.0.1:1234/v1",
)

# Model id из LM Studio /v1/models для nomic-embed-text-v1.5 GGUF.
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "text-embedding-nomic-embed-text-v1.5@q8_0",
)

# nomic-embed-text-v1.5 обычно даёт 768-мерные embeddings
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

TOP_K = int(os.getenv("TOP_K", "3"))


# === Тестовые документы ===

DOCUMENTS = [
    {
        "source": "note_qwen",
        "text": """
Qwen3.5-9B лучше использовать как генеративную модель в RAG.
Она получает найденные чанки текста и формирует финальный ответ для пользователя.
Сама чат-модель не должна использоваться для построения embedding-векторов.
""",
    },
    {
        "source": "note_embeddings",
        "text": """
Для RAG можно использовать nomic-embed-text-v1.5.
Документы эмбеддятся с префиксом search_document, а пользовательские запросы с префиксом search_query.
Полученные векторы можно хранить в pgvector.
""",
    },
    {
        "source": "note_pgvector",
        "text": """
pgvector — это расширение PostgreSQL для хранения и поиска векторов.
Для cosine similarity можно использовать оператор <=>.
Для ускорения поиска по embeddings можно создать HNSW индекс.
""",
    },
    {
        "source": "note_searxng",
        "text": """
SearXNG можно использовать как локальный метапоисковик.
Через MCP его можно подключить к LM Studio, чтобы локальная модель могла делать веб-поиск.
""",
    },
    {
        "source": "note_random",
        "text": """
Борщ — это суп со свёклой, капустой и мясом.
Рецепты борща могут отличаться в зависимости от региона.
""",
    },
]


# === LM Studio embedding client ===

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
)


def to_pgvector(vec: list[float]) -> str:
    """
    pgvector принимает строку вида:
    [0.1,0.2,0.3]
    """
    return "[" + ",".join(str(x) for x in vec) + "]"


def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )

    vec = response.data[0].embedding

    if len(vec) != EMBED_DIM:
        raise ValueError(
            f"Ожидал embedding размерности {EMBED_DIM}, "
            f"но модель вернула {len(vec)}. "
            f"Проверь EMBED_DIM или модель."
        )

    return vec


def embed_document(text: str) -> list[float]:
    # Важно для nomic
    return embed("search_document: " + text.strip())


def embed_query(text: str) -> list[float]:
    # Важно для nomic
    return embed("search_query: " + text.strip())


def check_embeddings() -> None:
    try:
        embed_query("healthcheck")
    except Exception as exc:
        raise RuntimeError(
            "Не удалось получить embedding от LM Studio. "
            f"Проверь, что server запущен на {LM_STUDIO_BASE_URL}, "
            f"а embedding-модель загружена с id '{EMBED_MODEL}'."
        ) from exc


def setup_db(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute("DROP TABLE IF EXISTS rag_chunks;")

        cur.execute(
            f"""
            CREATE TABLE rag_chunks (
                id bigserial PRIMARY KEY,
                content text NOT NULL,
                metadata jsonb DEFAULT '{{}}'::jsonb,
                embedding vector({EMBED_DIM}) NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE INDEX rag_chunks_embedding_hnsw
            ON rag_chunks
            USING hnsw (embedding vector_cosine_ops);
            """
        )

    conn.commit()


def insert_documents(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        for doc in DOCUMENTS:
            vec = embed_document(doc["text"])

            cur.execute(
                """
                INSERT INTO rag_chunks (content, metadata, embedding)
                VALUES (%s, %s, %s::vector);
                """,
                (
                    doc["text"].strip(),
                    Jsonb({"source": doc["source"]}),
                    to_pgvector(vec),
                ),
            )

    conn.commit()


def search(conn: psycopg.Connection, query: str, top_k: int = TOP_K):
    qvec = embed_query(query)
    qvec_pg = to_pgvector(qvec)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM rag_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (qvec_pg, qvec_pg, top_k),
        )

        return cur.fetchall()


def main():
    query = "Как использовать nomic embeddings вместе с pgvector для RAG?"

    print("Проверяю LM Studio embeddings...")
    check_embeddings()

    with psycopg.connect(DATABASE_URL) as conn:
        print("Создаю таблицу...")
        setup_db(conn)

        print("Эмбеддю и вставляю документы...")
        insert_documents(conn)

        print(f"\nЗапрос: {query}\n")
        results = search(conn, query)

        for idx, row in enumerate(results, start=1):
            doc_id, content, metadata, similarity = row

            print("=" * 80)
            print(f"RESULT #{idx}")
            print(f"id: {doc_id}")
            print(f"source: {metadata.get('source')}")
            print(f"similarity: {similarity:.4f}")
            print()
            print(content[:700])


if __name__ == "__main__":
    main()
