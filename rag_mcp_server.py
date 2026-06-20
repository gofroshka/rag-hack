import json
import os
import sys
import traceback
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER_NAME = "hack-rag"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"
RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
RAG_API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "30"))


def write_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def success(request_id: Any, result: dict[str, Any]) -> None:
    write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def error(request_id: Any, code: int, message: str, data: Any | None = None) -> None:
    payload: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        payload["data"] = data
    write_message({"jsonrpc": "2.0", "id": request_id, "error": payload})


def tools_list() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "search_rag",
                "description": (
                    "Search the hack-rag HTTP RAG server. "
                    "Use this when you need facts from the indexed Russian legal and ГОС2.0 documents."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query in Russian or English.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 5,
                            "description": "Number of chunks to return.",
                        },
                        "min_similarity": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                            "description": "Optional minimum cosine similarity threshold.",
                        },
                        "source_path": {
                            "type": "string",
                            "description": "Optional substring filter for document path/name.",
                        },
                        "content_width": {
                            "type": "integer",
                            "minimum": 200,
                            "maximum": 10000,
                            "default": 1400,
                            "description": "Maximum text width per returned chunk in the plain text summary.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "rag_stats",
                "description": "Return document and chunk counts from the hack-rag HTTP RAG server.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]
    }


def api_request(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{RAG_API_BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=RAG_API_TIMEOUT) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RAG API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to RAG API at {RAG_API_BASE_URL}. "
            "Start FastAPI or set RAG_API_BASE_URL."
        ) from exc

    return json.loads(response_body) if response_body else {}


def call_search_rag(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("Argument 'query' is required.")

    payload: dict[str, Any] = {
        "query": query,
        "top_k": int(arguments.get("top_k", 5)),
        "content_width": int(arguments.get("content_width", 1400)),
    }

    if arguments.get("min_similarity") is not None:
        payload["min_similarity"] = float(arguments["min_similarity"])
    if arguments.get("source_path"):
        payload["source_path"] = str(arguments["source_path"])

    response = api_request("/search", method="POST", payload=payload)
    text = response.get("answer_context") or "В RAG-индексе не найдено релевантных фрагментов."

    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": response,
        "isError": False,
    }


def call_rag_stats() -> dict[str, Any]:
    stats = api_request("/stats")
    text = "\n".join(
        [
            f"documents: {stats['documents']}",
            f"indexed_documents: {stats['indexed_documents']}",
            f"chunks: {stats['chunks']}",
        ]
    )
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": stats,
        "isError": False,
    }


def tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")

    if name == "search_rag":
        return call_search_rag(arguments)
    if name == "rag_stats":
        return call_rag_stats()

    raise ValueError(f"Unknown tool: {name}")


def handle_request(request: dict[str, Any]) -> None:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    is_notification = "id" not in request

    try:
        if method == "initialize":
            if is_notification:
                return
            success(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": (
                        "Use the search_rag tool to retrieve source snippets from the configured "
                        f"hack-rag HTTP server ({RAG_API_BASE_URL}) before answering questions "
                        "about the indexed dataset."
                    ),
                },
            )
            return

        if method in {"notifications/initialized", "notifications/cancelled"}:
            return

        if method == "ping":
            if not is_notification:
                success(request_id, {})
            return

        if method == "tools/list":
            if not is_notification:
                success(request_id, tools_list())
            return

        if method == "tools/call":
            if not is_notification:
                try:
                    success(request_id, tools_call(params))
                except Exception as exc:
                    success(
                        request_id,
                        {
                            "content": [{"type": "text", "text": f"RAG tool error: {exc}"}],
                            "isError": True,
                        },
                    )
            return

        if method == "resources/list":
            if not is_notification:
                success(request_id, {"resources": []})
            return

        if method == "prompts/list":
            if not is_notification:
                success(request_id, {"prompts": []})
            return

        if not is_notification:
            error(request_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        log(traceback.format_exc())
        if not is_notification:
            error(request_id, -32603, str(exc))


def handle_line(line: str) -> None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        error(None, -32700, f"Parse error: {exc}")
        return

    if isinstance(payload, list):
        for request in payload:
            if isinstance(request, dict):
                handle_request(request)
            else:
                error(None, -32600, "Invalid request")
        return

    if isinstance(payload, dict):
        handle_request(payload)
        return

    error(None, -32600, "Invalid request")


def main() -> None:
    log(f"{SERVER_NAME} MCP server started, RAG_API_BASE_URL={RAG_API_BASE_URL}")
    for line in sys.stdin:
        line = line.strip()
        if line:
            handle_line(line)


if __name__ == "__main__":
    main()
