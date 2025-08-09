# api/app.py
import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# ---------- Config (env) ----------
QDRANT_URL   = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://ollama:11434")
COLLECTION   = os.getenv("COLLECTION", "warframe")
TOP_K        = int(os.getenv("TOP_K", "8"))
EMBED_MODEL  = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GEN_MODEL    = os.getenv("GEN_MODEL", "llama3.1:8b-instruct-q4_K_M")
RAG_API_PORT = int(os.getenv("RAG_API_PORT", "8088"))

# ---------- Singletons ----------
_embedder = None
_qdrant: QdrantClient | None = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder

def embed(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
        # Make sure collection exists with the right vector size
        dim = len(embed(["probe"])[0])
        collections = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION not in collections:
            _qdrant.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    return _qdrant

# ---------- Retrieval ----------
def search(query: str, top_k: int) -> List[Dict[str, Any]]:
    vec = embed([query])[0]
    res = get_qdrant().search(
        collection_name=COLLECTION,
        query_vector=vec,
        with_payload=True,
        limit=top_k,
    )
    out: List[Dict[str, Any]] = []
    for p in res:
        payload = p.payload or {}
        out.append({
            "id": p.id,
            "score": p.score,
            "url": payload.get("url"),
            "title": payload.get("title"),
            "text": payload.get("text"),
        })
    return out

def build_prompt(query: str, hits: List[Dict[str, Any]]):
    ctx_blocks, cites = [], []
    for h in hits:
        if not h.get("text"):
            continue
        ctx_blocks.append(
            f"Source: {h.get('title')}\nURL: {h.get('url')}\n---\n{h.get('text')}"
        )
        cites.append({"title": h.get("title"), "url": h.get("url")})
    context = "\n\n".join(ctx_blocks)
    prompt = (
        "You are a Warframe expert. Answer ONLY using the sources below. "
        "Cite inline as [1], [2], ... matching order. If unknown, say you don't know.\n\n"
        f"Question: {query}\n\nSOURCES:\n{context}\n\nAnswer:"
    )
    # de-dup citations, preserving order
    seen = set(); uniq = []
    for c in cites:
        key = (c["title"], c["url"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return prompt, uniq

# ---------- Generation ----------
async def ollama_chat(prompt: str, model: str = GEN_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer strictly from provided context with citations."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_ctx": 4096}
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

# ---------- API ----------
app = FastAPI(title="Warframe RAG API", version="0.1.0")

class AskRequest(BaseModel):
    query: str
    top_k: int | None = None
    model: str | None = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    # light checks
    _ = get_embedder()
    _ = get_qdrant()
    return {"ready": True}

@app.post("/ask")
async def ask(req: AskRequest):
    k = req.top_k or TOP_K
    hits = search(req.query.strip(), k)
    if not hits:
        raise HTTPException(status_code=503, detail="No indexed context yet. Run ingestor.")
    prompt, cites = build_prompt(req.query, hits)
    answer = await ollama_chat(prompt, model=(req.model or GEN_MODEL))
    return {"answer": answer, "sources": cites}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=RAG_API_PORT)
