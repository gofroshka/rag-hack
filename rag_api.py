from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag_dataset import (
    EMBED_MODEL,
    LM_STUDIO_BASE_URL,
    format_results_for_llm,
    get_rag_stats,
    search_rag,
)


app = FastAPI(
    title="hack-rag API",
    description="Local FastAPI server for searching the indexed RAG dataset.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query.")
    top_k: int = Field(5, ge=1, le=50)
    min_similarity: float | None = Field(None, ge=-1, le=1)
    source_path: str | None = Field(None, description="Optional substring filter for source_path.")
    content_width: int | None = Field(
        None,
        ge=200,
        le=10000,
        description="When set, shorten each result content to this width.",
    )


def run_search(request: SearchRequest) -> dict:
    try:
        results = search_rag(
            request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            source_path=request.source_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "query": request.query,
        "top_k": request.top_k,
        "count": len(results),
        "answer_context": format_results_for_llm(results, width=request.content_width or 1400),
        "results": [
            result.to_dict(content_width=request.content_width)
            for result in results
        ],
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "hack-rag API",
        "routes": ["/health", "/stats", "/search"],
    }


@app.get("/health")
def health() -> dict:
    try:
        stats = get_rag_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "lm_studio_base_url": LM_STUDIO_BASE_URL,
        "embed_model": EMBED_MODEL,
        "documents": stats["documents"],
        "indexed_documents": stats["indexed_documents"],
        "chunks": stats["chunks"],
    }


@app.get("/stats")
def stats() -> dict:
    try:
        return get_rag_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
def search_post(request: SearchRequest) -> dict:
    return run_search(request)


@app.get("/search")
def search_get(
    query: Annotated[str, Query(min_length=1)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
    min_similarity: Annotated[float | None, Query(ge=-1, le=1)] = None,
    source_path: str | None = None,
    content_width: Annotated[int | None, Query(ge=200, le=10000)] = None,
) -> dict:
    return run_search(
        SearchRequest(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            source_path=source_path,
            content_width=content_width,
        )
    )
