import argparse
import asyncio
import hashlib
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import httpx
import orjson
from fastapi import FastAPI, Body
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from bs4 import BeautifulSoup
from readability import Document
import trafilatura

COLLECTION = os.getenv("COLLECTION", "warframe")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", 8))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
RECENCY_BOOST_DAYS = int(os.getenv("RECENCY_BOOST_DAYS", 150))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.1:8b-instruct-q4_K_M")

_embedder = None
_qdrant = None

# ------------------ Embeddings ------------------

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _embedder


def embed_texts(texts: List[str]):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()

# ------------------ Qdrant ------------------

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
        dim = len(embed_texts(["test"])[0])
        collections = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION not in collections:
            _qdrant.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    return _qdrant

# ------------------ Utils ------------------

def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def stable_id(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:12], 16)


def clean_html(html: str) -> str:
    # Try trafilatura first; fallback to readability + BS4
    try:
        text = trafilatura.extract(html, include_tables=True, include_images=False)
        if text and len(text.split()) > 50:
            return text
    except Exception:
        pass
    try:
        doc = Document(html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text("\n")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Approximate token-based chunking by characters
    if not text:
        return []
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size*4]  # ~4 chars per token heuristic
        chunks.append(chunk)
        i += size*4 - overlap*4
    return chunks


async def http_get(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        r = await client.get(url, headers={"User-Agent": "warframe-rag/1.0"})
        r.raise_for_status()
        return r.text


def extract_links(html: str, base_domain_allow: List[str]) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/") and base_domain_allow:
            # We don't know scheme/host here; caller should normalize if needed.
            out.append(href)
        elif href.startswith("http"):
            out.append(href)
    return out


# ------------------ Ingestion ------------------

def allowed(url: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    return any(p in url for p in patterns)


async def ingest_sources(sources_path: str, rules_path: str):
    import yaml

    with open(sources_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rules = {}
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}

    qdrant = get_qdrant()

    for coll in cfg.get("collections", []):
        name = coll.get("name")
        max_pages = int(coll.get("max_pages", 500))
        allow_patterns = coll.get("allow_patterns", [])
        seed_urls = coll.get("urls", [])

        print(f"[ingest] Collection {name}: seeds={len(seed_urls)} max_pages={max_pages}")
        seen = set()
        queue = list(seed_urls)
        pages = 0

        async with httpx.AsyncClient(follow_redirects=True, timeout=45) as client:
            while queue and pages < max_pages:
                url = queue.pop(0)
                if url in seen:
                    continue
                if not allowed(url, allow_patterns):
                    continue
                seen.add(url)
                try:
                    resp = await client.get(url, headers={"User-Agent": "warframe-rag/1.0"})
                    resp.raise_for_status()
                    html = resp.text
                except Exception as e:
                    print(f"[ingest] FAIL {url}: {e}")
                    continue

                text = clean_html(html)
                if not text or len(text.split()) < 60:
                    continue

                title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
                title = title_match.group(1).strip() if title_match else url
                chunks = chunk_text(text)
                vecs = embed_texts(chunks)

                points = []
                for i, (chunk, v) in enumerate(zip(chunks, vecs)):
                    pid = stable_id(f"{url}#chunk-{i}")
                    payload = {
                        "url": url,
                        "title": title,
                        "source_collection": name,
                        "ingested_at": now_iso(),
                    }
                    points.append(PointStruct(id=pid, vector=v, payload=payload))

                try:
                    qdrant.upsert(collection_name=COLLECTION, points=points)
                    pages += 1
                except Exception as e:
                    print(f"[ingest] Qdrant upsert error at {url}: {e}")

                # expand links (simple heuristic, throttled)
                if len(seen) < max_pages:
                    for link in extract_links(html, []):
                        if link.startswith("/"):
                            # best-effort host derivation
                            from urllib.parse import urlparse, urljoin
                            base = urlparse(url)
                            absu = urljoin(f"{base.scheme}://{base.netloc}", link)
                            if allowed(absu, allow_patterns):
                                queue.append(absu)
                        else:
                            if allowed(link, allow_patterns):
                                queue.append(link)

                if pages % 20 == 0:
                    print(f"[ingest] {name}: pages indexed {pages}")

    print("[ingest] complete")

# ------------------ Retrieval & Generation ------------------

class AskRequest(BaseModel):
    query: str
    top_k: int | None = None


async def ollama_generate(prompt: str, model: str = GEN_MODEL, system: str | None = None) -> str:
    # Use chat API for better formatting
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or "You are a Warframe expert. Answer ONLY using the provided context and cite sources as [1], [2], etc. If unknown, say you don't know."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_ctx": 4096}
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        # OpenAI-compatible route returns choices[0].message.content
        return data["choices"][0]["message"]["content"]


def search_qdrant(query: str, top_k: int) -> List[Dict[str, Any]]:
    qdrant = get_qdrant()
    qvec = embed_texts([query])[0]

    # Recency booster via filter on ingested_at (optional)
    flt = None
    if RECENCY_BOOST_DAYS > 0:
        cutoff = (datetime.utcnow() - timedelta(days=RECENCY_BOOST_DAYS)).isoformat() + "Z"
        flt = Filter(
            must=[
                FieldCondition(key="ingested_at", match=MatchValue(value=None))
            ]
        )
        # Note: Qdrant doesn't natively filter ISO strings by range in basic client.
        # For simplicity, we skip range filtering here. A real booster would use a scalar timestamp field.

    res = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        with_payload=True,
        limit=top_k,
        score_threshold=None
    )

    out = []
    for point in res:
        payload = point.payload or {}
        out.append({
            "score": point.score,
            "text": None,  # text is not stored; we rely on payload metadata + context in prompt
            "url": payload.get("url"),
            "title": payload.get("title"),
            "ingested_at": payload.get("ingested_at"),
            "id": point.id,
        })
    return out


def fetch_context_snippets(ids: List[int], window: int = 2) -> List[Dict[str, Any]]:
    # Re-fetch neighbors around found chunks for richer context
    qdrant = get_qdrant()
    out = []
    for pid in ids:
        # nearest neighbors to the same point id (approximate sequential chunks)
        try:
            res = qdrant.recommend(
                collection_name=COLLECTION,
                positive=[pid],
                limit=window,
                with_payload=True
            )
            for p in res:
                out.append({
                    "url": p.payload.get("url"),
                    "title": p.payload.get("title"),
                    "id": p.id
                })
        except Exception:
            pass
    # de-dup by (url, title)
    uniq = {}
    for x in out:
        uniq[(x["url"], x["title"])]=x
    return list(uniq.values())


def build_prompt(query: str, hits: List[Dict[str, Any]]):
    # Pull raw text for context? We didn't store text to save space.
    # For simplicity, we will do another lightweight search-by-id using scroll (not ideal, but ok for MVP),
    # and include URLs + titles as citations anchors. The model should answer based on its knowledge guided by citations.
    # Better approach: store chunk text in payload. Let's do that now for accuracy.
    pass

# Let's redefine upsert to also store text in payload (update above where points created) and complete the server.
# (You already saw upsert creation earlier; here's a helper we can also use to backfill.)

from qdrant_client.http.models import PointStruct


def upsert_chunks(url: str, title: str, chunks: List[str], vectors: List[List[float]], collection: str = COLLECTION):
    qdrant = get_qdrant()
    points = []
    for i, (chunk, v) in enumerate(zip(chunks, vectors)):
        pid = stable_id(f"{url}#chunk-{i}")
        payload = {
            "url": url,
            "title": title,
            "ingested_at": now_iso(),
            "text": chunk,
        }
        points.append(PointStruct(id=pid, vector=v, payload=payload))
    qdrant.upsert(collection_name=collection, points=points)


# Patch ingest_sources to use upsert_chunks with text payload
async def ingest_sources(sources_path: str, rules_path: str):
    import yaml

    with open(sources_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qdrant = get_qdrant()

    for coll in cfg.get("collections", []):
        name = coll.get("name")
        max_pages = int(coll.get("max_pages", 500))
        allow_patterns = coll.get("allow_patterns", [])
        seed_urls = coll.get("urls", [])

        print(f"[ingest] Collection {name}: seeds={len(seed_urls)} max_pages={max_pages}")
        seen = set()
        queue = list(seed_urls)
        pages = 0

        async with httpx.AsyncClient(follow_redirects=True, timeout=45) as client:
            while queue and pages < max_pages:
                url = queue.pop(0)
                if url in seen:
                    continue
                if not allowed(url, allow_patterns):
                    continue
                seen.add(url)
                try:
                    resp = await client.get(url, headers={"User-Agent": "warframe-rag/1.0"})
                    resp.raise_for_status()
                    html = resp.text
                except Exception as e:
                    print(f"[ingest] FAIL {url}: {e}")
                    continue

                text = clean_html(html)
                if not text or len(text.split()) < 60:
                    continue

                title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
                title = title_match.group(1).strip() if title_match else url
                chunks = chunk_text(text)
                vecs = embed_texts(chunks)

                try:
                    upsert_chunks(url, title, chunks, vecs, collection=COLLECTION)
                    pages += 1
                except Exception as e:
                    print(f"[ingest] Qdrant upsert error at {url}: {e}")

                # expand links
                if len(seen) < max_pages:
                    for link in extract_links(html, []):
                        if link.startswith("/"):
                            from urllib.parse import urlparse, urljoin
                            base = urlparse(url)
                            absu = urljoin(f"{base.scheme}://{base.netloc}", link)
                            if allowed(absu, allow_patterns):
                                queue.append(absu)
                        else:
                            if allowed(link, allow_patterns):
                                queue.append(link)

                if pages % 20 == 0:
                    print(f"[ingest] {name}: pages indexed {pages}")

    print("[ingest] complete")


# --------- Retrieval prompt construction ---------

def build_prompt(query: str, hits: List[Dict[str, Any]]):
    # Grab the text payloads for top hits
    qdrant = get_qdrant()
    ctxs = []
    citations = []

    for h in hits:
        try:
            p = qdrant.retrieve(collection_name=COLLECTION, ids=[h["id"]], with_payload=True)
            if p:
                payload = p[0].payload or {}
                txt = payload.get("text")
                url = payload.get("url")
                title = payload.get("title")
                if txt:
                    ctxs.append(f"Source: {title}\nURL: {url}\n---\n{txt}")
                    citations.append({"title": title, "url": url})
        except Exception:
            continue

    context_block = "\n\n".join(ctxs[:TOP_K])
    prompt = (
        "You are a Warframe expert. Use ONLY the following sources to answer. "
        "Cite sources inline as [1], [2], etc., matching the order provided below. If insufficient info, say you don't know.\n\n"
        f"Question: {query}\n\n"
        f"SOURCES (ordered):\n{context_block}\n\n"
        "Answer:"
    )
    return prompt, citations


# ------------------ FastAPI ------------------

app = FastAPI(title="Warframe RAG API", version="0.1.0")


@app.post("/ask")
async def ask(req: AskRequest):
    query = req.query.strip()
    k = req.top_k or TOP_K
    hits = search_qdrant(query, k)
    if not hits:
        return {"answer": "I don't have enough Warframe context indexed yet.", "sources": []}

    prompt, citations = build_prompt(query, hits)
    answer = await ollama_generate(prompt, model=GEN_MODEL)
    # Deduplicate citations preserving order
    seen = set()
    unique_cites = []
    for c in citations:
        key = (c["title"], c["url"])
        if key not in seen:
            seen.add(key)
            unique_cites.append(c)
    return {"answer": answer, "sources": unique_cites}


# ------------------ CLI ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", type=str, help="Path to sources.yaml")
    parser.add_argument("--rules", type=str, default=None, help="Path to rules.yaml")
    args = parser.parse_args()

    if args.ingest:
        asyncio.run(ingest_sources(args.ingest, args.rules))
    else:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("RAG_API_PORT", 8088)))