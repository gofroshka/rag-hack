import argparse
import hashlib
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import psycopg
from openai import OpenAI
from psycopg.types.json import Jsonb


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/ragdb",
)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "text-embedding-nomic-embed-text-v1.5@q8_0",
)
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

SUPPORTED_TEXT_SUFFIXES = {".docx", ".pdf"}
SKIPPED_SUFFIXES = {".mp4"}

DOCS_TABLE = "rag_dataset_documents"
CHUNKS_TABLE = "rag_dataset_chunks"

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")


@dataclass
class RagSearchResult:
    rank: int
    chunk_id: int
    source_name: str
    source_path: str
    chunk_index: int
    content: str
    similarity: float

    def to_dict(self, *, content_width: int | None = None) -> dict:
        content = self.content
        if content_width is not None:
            content = make_excerpt(content, content_width)

        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "similarity": self.similarity,
            "content": content,
        }


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embeddings_by_index = {item.index: item.embedding for item in response.data}
    embeddings = [embeddings_by_index[i] for i in range(len(texts))]

    for vec in embeddings:
        if len(vec) != EMBED_DIM:
            raise ValueError(
                f"Ожидал embedding размерности {EMBED_DIM}, но модель вернула {len(vec)}. "
                "Проверь EMBED_DIM или модель."
            )

    return embeddings


def embed_query(query: str) -> list[float]:
    return embed_texts(["search_query: " + query.strip()])[0]


def setup_db(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DOCS_TABLE} (
                id bigserial PRIMARY KEY,
                source_path text NOT NULL UNIQUE,
                source_name text NOT NULL,
                file_type text NOT NULL,
                sha256 text NOT NULL,
                chunk_count integer NOT NULL DEFAULT 0,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                indexed_at timestamptz
            );
            """
        )
        cur.execute(
            f"ALTER TABLE {DOCS_TABLE} ADD COLUMN IF NOT EXISTS chunk_count integer NOT NULL DEFAULT 0;"
        )
        cur.execute(
            f"ALTER TABLE {DOCS_TABLE} ADD COLUMN IF NOT EXISTS indexed_at timestamptz;"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
                id bigserial PRIMARY KEY,
                document_id bigint NOT NULL REFERENCES {DOCS_TABLE}(id) ON DELETE CASCADE,
                chunk_index integer NOT NULL,
                content text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({EMBED_DIM}) NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (document_id, chunk_index)
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {CHUNKS_TABLE}_embedding_hnsw
            ON {CHUNKS_TABLE}
            USING hnsw (embedding vector_cosine_ops);
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {CHUNKS_TABLE}_document_id_idx
            ON {CHUNKS_TABLE} (document_id);
            """
        )

    conn.commit()


def clamp_top_k(top_k: int, maximum: int = 50) -> int:
    return max(1, min(top_k, maximum))


def make_excerpt(content: str, width: int = 900) -> str:
    return textwrap.shorten(" ".join(content.split()), width=width, placeholder="...")


def search_rag(
    query: str,
    *,
    top_k: int = 5,
    min_similarity: float | None = None,
    source_path: str | None = None,
) -> list[RagSearchResult]:
    top_k = clamp_top_k(top_k)
    query_vec = to_pgvector(embed_query(query))
    params: list[object] = [query_vec, query_vec]
    filters = ["d.indexed_at IS NOT NULL"]

    if source_path:
        filters.append("d.source_path ILIKE %s")
        params.append(f"%{source_path}%")

    where_sql = " AND ".join(filters)
    fetch_limit = top_k if min_similarity is None else min(top_k * 5, 100)
    params.append(fetch_limit)

    with psycopg.connect(DATABASE_URL) as conn:
        setup_db(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    c.id,
                    d.source_name,
                    d.source_path,
                    c.chunk_index,
                    c.content,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM {CHUNKS_TABLE} c
                JOIN {DOCS_TABLE} d ON d.id = c.document_id
                WHERE {where_sql}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                params,
            )
            rows = cur.fetchall()

    results: list[RagSearchResult] = []
    for row in rows:
        chunk_id, source_name, source_path_value, chunk_index, content, similarity = row
        similarity = float(similarity)
        if min_similarity is not None and similarity < min_similarity:
            continue
        results.append(
            RagSearchResult(
                rank=len(results) + 1,
                chunk_id=chunk_id,
                source_name=source_name,
                source_path=source_path_value,
                chunk_index=chunk_index,
                content=content,
                similarity=similarity,
            )
        )
        if len(results) >= top_k:
            break

    return results


def get_rag_stats() -> dict:
    with psycopg.connect(DATABASE_URL) as conn:
        setup_db(conn)
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {DOCS_TABLE};")
            documents = cur.fetchone()[0]
            cur.execute(f"SELECT count(*) FROM {CHUNKS_TABLE};")
            chunks = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT count(*)
                FROM {DOCS_TABLE}
                WHERE indexed_at IS NOT NULL;
                """
            )
            indexed_documents = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT d.source_path, count(c.id) AS chunks
                FROM {DOCS_TABLE} d
                LEFT JOIN {CHUNKS_TABLE} c ON c.document_id = d.id
                GROUP BY d.source_path
                ORDER BY d.source_path;
                """
            )
            sources = [
                {"source_path": source_path, "chunks": chunk_count}
                for source_path, chunk_count in cur.fetchall()
            ]

    return {
        "documents": documents,
        "indexed_documents": indexed_documents,
        "chunks": chunks,
        "sources": sources,
    }


def format_results_for_llm(results: list[RagSearchResult], *, width: int = 1400) -> str:
    if not results:
        return "В RAG-индексе не найдено релевантных фрагментов."

    parts: list[str] = []
    for result in results:
        parts.append(
            "\n".join(
                [
                    f"[{result.rank}] {result.source_name}",
                    f"path: {result.source_path}",
                    f"chunk: {result.chunk_index}",
                    f"similarity: {result.similarity:.4f}",
                    make_excerpt(result.content, width),
                ]
            )
        )
    return "\n\n".join(parts)


def reset_dataset(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {DOCS_TABLE} RESTART IDENTITY CASCADE;")
    conn.commit()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def extract_pdf_text(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Для PDF нужен Poppler CLI `pdftotext`. Установи его или конвертируй PDF в DOCX/TXT."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Не удалось извлечь текст из PDF: {path}") from exc

    return result.stdout


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return normalize_text(extract_docx_text(path))
    if suffix == ".pdf":
        return normalize_text(extract_pdf_text(path))
    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if overlap >= max_chars:
        raise ValueError("--overlap должен быть меньше --chunk-size")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_current()
            start = 0
            while start < len(paragraph):
                end = start + max_chars
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap)
            continue

        next_len = current_len + len(paragraph) + (2 if current else 0)
        if next_len > max_chars:
            flush_current()

        current.append(paragraph)
        current_len += len(paragraph) + (2 if len(current) > 1 else 0)

    flush_current()

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    previous_tail = ""
    for chunk in chunks:
        combined = f"{previous_tail}\n\n{chunk}".strip() if previous_tail else chunk
        overlapped.append(combined[-max_chars:])
        previous_tail = chunk[-overlap:]

    return overlapped


def iter_dataset_files(dataset_dir: Path) -> list[Path]:
    files = [path for path in dataset_dir.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: str(path).casefold())


def relative_source_path(path: Path, dataset_dir: Path) -> str:
    return path.relative_to(dataset_dir).as_posix()


def is_document_complete(
    conn: psycopg.Connection,
    source_path: str,
    sha256: str,
    expected_chunks: int,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.sha256, d.chunk_count, d.indexed_at, count(c.id)
            FROM {DOCS_TABLE} d
            LEFT JOIN {CHUNKS_TABLE} c ON c.document_id = d.id
            WHERE d.source_path = %s
            GROUP BY d.id;
            """,
            (source_path,),
        )
        row = cur.fetchone()
    return bool(
        row
        and row[0] == sha256
        and row[1] == expected_chunks
        and row[2] is not None
        and row[3] == expected_chunks
    )


def delete_existing_document(conn: psycopg.Connection, source_path: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {DOCS_TABLE} WHERE source_path = %s;", (source_path,))
    conn.commit()


def create_document(
    conn: psycopg.Connection,
    source_path: str,
    path: Path,
    file_hash: str,
    chunk_count: int,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {DOCS_TABLE} (source_path, source_name, file_type, sha256, chunk_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                source_path,
                path.name,
                path.suffix.lower().lstrip("."),
                file_hash,
                chunk_count,
                Jsonb({"size_bytes": path.stat().st_size, "chunk_count": chunk_count}),
            ),
        )
        document_id = cur.fetchone()[0]

    conn.commit()
    return document_id


def insert_chunk_batch(
    conn: psycopg.Connection,
    document_id: int,
    source_path: str,
    start_index: int,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    with conn.cursor() as cur:
        for offset, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_index = start_index + offset
            cur.execute(
                f"""
                INSERT INTO {CHUNKS_TABLE} (document_id, chunk_index, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::vector);
                """,
                (
                    document_id,
                    chunk_index,
                    chunk,
                    Jsonb({"source_path": source_path, "chunk_index": chunk_index}),
                    to_pgvector(embedding),
                ),
            )

    conn.commit()


def mark_document_indexed(conn: psycopg.Connection, document_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {DOCS_TABLE}
            SET indexed_at = now(), updated_at = now()
            WHERE id = %s;
            """,
            (document_id,),
        )
    conn.commit()


def index_chunks_in_batches(
    conn: psycopg.Connection,
    document_id: int,
    source_path: str,
    chunks: list[str],
) -> None:
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_chunks = chunks[start : start + EMBED_BATCH_SIZE]
        prepared = ["search_document: " + chunk for chunk in batch_chunks]
        embeddings = embed_texts(prepared)
        insert_chunk_batch(conn, document_id, source_path, start, batch_chunks, embeddings)
        print(
            f"  indexed chunks: {min(start + len(batch_chunks), len(chunks))}/{len(chunks)}",
            flush=True,
        )


def ingest_dataset(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {dataset_dir}")

    files = iter_dataset_files(dataset_dir)
    if args.limit_files:
        files = files[: args.limit_files]

    conn: psycopg.Connection | None = None
    try:
        if args.dry_run:
            if args.reset:
                print("dry-run: database reset skipped", flush=True)
        else:
            conn = psycopg.connect(DATABASE_URL)
            setup_db(conn)
            if args.reset:
                reset_dataset(conn)

        indexed = 0
        skipped = 0
        failed = 0
        total_chunks = 0

        for path in files:
            suffix = path.suffix.lower()
            source_path = relative_source_path(path, dataset_dir)

            if suffix in SKIPPED_SUFFIXES:
                print(f"SKIP unsupported media: {source_path}", flush=True)
                skipped += 1
                continue
            if suffix not in SUPPORTED_TEXT_SUFFIXES:
                print(f"SKIP unsupported file: {source_path}", flush=True)
                skipped += 1
                continue

            print(f"\nFILE {source_path}", flush=True)
            try:
                file_hash = sha256_file(path)
                text = extract_text(path)
                chunks = chunk_text(text, args.chunk_size, args.overlap)

                if not chunks:
                    print("  empty text after extraction", flush=True)
                    skipped += 1
                    continue

                print(f"  chars: {len(text)}", flush=True)
                print(f"  chunks: {len(chunks)}", flush=True)

                if args.dry_run:
                    indexed += 1
                    total_chunks += len(chunks)
                    continue

                if conn is None:
                    raise RuntimeError("Database connection is not initialized")

                if not args.force and is_document_complete(conn, source_path, file_hash, len(chunks)):
                    print("  already indexed", flush=True)
                    skipped += 1
                    continue

                delete_existing_document(conn, source_path)
                document_id = create_document(conn, source_path, path, file_hash, len(chunks))
                index_chunks_in_batches(conn, document_id, source_path, chunks)
                mark_document_indexed(conn, document_id)
                indexed += 1
                total_chunks += len(chunks)
            except Exception as exc:
                failed += 1
                print(f"  ERROR: {exc}", file=sys.stderr, flush=True)

        print("\nDONE", flush=True)
        print(f"indexed files: {indexed}", flush=True)
        print(f"chunks: {total_chunks}", flush=True)
        print(f"skipped files: {skipped}", flush=True)
        print(f"failed files: {failed}", flush=True)
    finally:
        if conn is not None:
            conn.close()


def search_dataset(args: argparse.Namespace) -> None:
    results = search_rag(args.query, top_k=args.top_k)

    if not results:
        print("Ничего не найдено. Сначала запусти ingest.")
        return

    for result in results:
        excerpt = make_excerpt(result.content, args.width)
        print("=" * 100)
        print(f"RESULT #{result.rank}")
        print(f"source: {result.source_name}")
        print(f"path: {result.source_path}")
        print(f"chunk: {result.chunk_index}")
        print(f"similarity: {result.similarity:.4f}")
        print()
        print(excerpt)


def stats(args: argparse.Namespace) -> None:
    data = get_rag_stats()
    print(f"documents: {data['documents']}")
    print(f"indexed documents: {data['indexed_documents']}")
    print(f"chunks: {data['chunks']}")
    for source in data["sources"]:
        print(f"{source['chunks']:5}  {source['source_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Индексация и поиск по локальному RAG-датасету.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Извлечь текст, нарезать на чанки и записать embeddings в БД.")
    ingest.add_argument("--dataset-dir", type=Path, default=Path("датасет для rag"))
    ingest.add_argument("--chunk-size", type=int, default=1800)
    ingest.add_argument("--overlap", type=int, default=250)
    ingest.add_argument("--reset", action="store_true", help="Очистить индекс датасета перед загрузкой.")
    ingest.add_argument("--force", action="store_true", help="Переиндексировать файлы даже если sha256 не изменился.")
    ingest.add_argument("--dry-run", action="store_true", help="Только извлечь текст и посчитать чанки без embeddings.")
    ingest.add_argument("--limit-files", type=int, help="Обработать только первые N файлов для быстрой проверки.")
    ingest.set_defaults(func=ingest_dataset)

    search = subparsers.add_parser("search", help="Найти релевантные чанки в индексированном датасете.")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--width", type=int, default=900)
    search.set_defaults(func=search_dataset)

    stats_parser = subparsers.add_parser("stats", help="Показать статистику индексированного датасета.")
    stats_parser.set_defaults(func=stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
