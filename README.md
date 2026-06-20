# hack-rag

Минимальный локальный RAG smoke test:

- embeddings берутся из LM Studio через OpenAI-compatible API;
- векторы хранятся в PostgreSQL с расширением pgvector;
- `main.py` пересоздает таблицу `rag_chunks`, вставляет тестовые документы и выполняет similarity search.

## Запуск базы

```bash
docker compose up -d postgres
```

База доступна по умолчанию здесь:

```text
postgresql://postgres:postgres@127.0.0.1:5432/ragdb
```

## Запуск LM Studio

В LM Studio включи локальный OpenAI-compatible server на `http://127.0.0.1:1234/v1` и загрузи embedding-модель `nomic-ai/nomic-embed-text-v1.5-GGUF`.

Проверенный локальный model id:

```bash
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0 uv run python main.py
```

Если в LM Studio используется другая квантизация, посмотри доступные id:

```bash
curl http://127.0.0.1:1234/v1/models
```

## Проверка RAG-скрипта

```bash
uv run python main.py
```

Настройки можно переопределить переменными окружения:

```bash
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/ragdb \
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1 \
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0 \
uv run python main.py
```

## Индексация датасета

Скрипт `rag_dataset.py` обрабатывает файлы из папки `датасет для rag`, пишет документы в таблицы `rag_dataset_documents` и `rag_dataset_chunks`, а затем позволяет искать по ним.

Поддерживаются:

- `.docx` через встроенный parser;
- `.pdf` через `pdftotext` из Poppler;
- `.mp4` сейчас пропускается, для него нужна отдельная транскрибация в текст.

Проверить парсинг без записи embeddings:

```bash
uv run python rag_dataset.py ingest --dry-run
```

Полностью переиндексировать датасет:

```bash
uv run python rag_dataset.py ingest --reset
```

Посмотреть статистику:

```bash
uv run python rag_dataset.py stats
```

Искать по датасету:

```bash
uv run python rag_dataset.py search "Какие документы нужны заказчику по 44-ФЗ?" --top-k 5
```

Если хочешь ускорить или замедлить нагрузку на LM Studio, меняй размер batch:

```bash
EMBED_BATCH_SIZE=32 uv run python rag_dataset.py ingest --reset
```

## FastAPI RAG server

Запуск HTTP API:

```bash
uv run uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
```

Поиск через GET:

```bash
curl "http://127.0.0.1:8000/search?query=Какие%20документы%20нужны%20заказчику%20по%2044-ФЗ%3F&top_k=3&content_width=700"
```

Поиск через POST:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Какие документы нужны заказчику по 44-ФЗ?",
    "top_k": 3,
    "content_width": 700
  }'
```

## MCP server для LM Studio

MCP сервер работает по stdio, но сам RAG не выполняет. Он дергает FastAPI RAG server по HTTP (`RAG_API_BASE_URL`) и отдает Qwen инструменты:

- `search_rag` — поиск релевантных чанков в RAG;
- `rag_stats` — статистика индекса.

Если LM Studio/Qwen и RAG server на одной машине:

```bash
uv run uvicorn rag_api:app --host 127.0.0.1 --port 8000
uv run python rag_mcp_server.py
```

Если MCP запускается на другой машине, подними API на сетевом интерфейсе RAG-сервера:

```bash
uv run uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

И укажи URL RAG-сервера:

```bash
RAG_API_BASE_URL=http://<rag-server-ip>:8000 uv run python rag_mcp_server.py
```

Пример MCP-конфига для LM Studio:

```json
{
  "mcpServers": {
    "hack-rag": {
      "command": "/Users/georgiy/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/georgiy/Dev/hack-rag",
        "run",
        "python",
        "rag_mcp_server.py"
      ],
      "env": {
        "RAG_API_BASE_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

Перед подключением MCP проверь, что запущены:

- Docker база: `docker compose up -d postgres`
- LM Studio server: `http://127.0.0.1:1234/v1`
- embedding model: `text-embedding-nomic-embed-text-v1.5@q8_0`
- FastAPI RAG server: `uv run uvicorn rag_api:app --host 127.0.0.1 --port 8000`
