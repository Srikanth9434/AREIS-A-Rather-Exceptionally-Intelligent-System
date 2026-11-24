"""
rag_engine.py
Offline RAG + vector store for AREIS.

Persists next to the exe (or this script) in rag_db/store.sqlite so knowledge
survives restarts and works when frozen. Each INDUSTRY / EMPLOYEE context is
its own collection.

Embeddings:
  1. llama-server /v1/embeddings (same local Qwen process, GPU)
  2. deterministic hashed n-gram vectors if the embeddings endpoint is missing

No extra model files and no internet. Retrieval is cosine similarity over
numpy vectors loaded from SQLite.

File types: plain text (.txt .md .csv .json .log .tsv), plus PDF (.pdf) and
Word (.docx) via pypdf / python-docx -- both text-layer extraction only, so
scanned/image-only PDFs won't yield usable text (they'd need OCR first).
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from typing import Optional

import numpy as np
import requests

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR = os.path.join(BASE_DIR, "rag_db")
DB_PATH = os.path.join(DB_DIR, "store.sqlite")
EMBED_URL_OPENAI = "http://127.0.0.1:8080/v1/embeddings"
EMBED_URL_NATIVE = "http://127.0.0.1:8080/embedding"

CHUNK_CHARS = 700
CHUNK_OVERLAP = 120
HASH_DIM = 384
TOP_K = 4

VALID_COLLECTIONS = ("industry", "employee")
TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".tsv"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
ALL_EXTS = TEXT_EXTS | PDF_EXTS | DOCX_EXTS


def _connect():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection)"
    )
    conn.commit()
    return conn


def _meta_get(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _meta_set(conn, key, value):
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (key, str(value)),
    )


def _hash_embed(text: str, dim: int = HASH_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    grams = list(tokens)
    grams += [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _as_vector(emb) -> np.ndarray:
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    arr = np.ravel(arr)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


_llama_ok = None


def _llama_embed(text: str) -> Optional[np.ndarray]:
    global _llama_ok
    if _llama_ok is False:
        return None

    try:
        r = requests.post(
            EMBED_URL_OPENAI,
            json={"input": text, "encoding_format": "float"},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            _llama_ok = True
            return _as_vector(data["data"][0]["embedding"])
    except Exception:
        pass

    try:
        r = requests.post(
            EMBED_URL_NATIVE,
            json={"content": text},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            emb = data.get("embedding", data)
            _llama_ok = True
            return _as_vector(emb)
    except Exception:
        pass

    _llama_ok = False
    return None


def embed_text(text: str) -> np.ndarray:
    conn = _connect()
    backend = _meta_get(conn, "backend")
    dim = _meta_get(conn, "dim")
    conn.close()

    if backend == "hash":
        return _hash_embed(text, dim=int(dim) if dim else HASH_DIM)

    vec = _llama_embed(text)
    if vec is not None:
        if dim is None or int(dim) == int(vec.shape[0]):
            return vec

    if backend == "llama":
        raise RuntimeError(
            "Vector store was built with llama-server embeddings, but the "
            "embeddings endpoint is not responding."
        )
    return _hash_embed(text, dim=HASH_DIM)


def ingest_text(text: str, collection: str, source: str = "pasted") -> dict:
    collection = collection.lower()
    if collection not in VALID_COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection}")

    pieces = chunk_text(text)
    if not pieces:
        return {"added": 0, "source": source, "collection": collection}

    conn = _connect()
    added = 0
    for piece in pieces:
        backend = _meta_get(conn, "backend")
        llama_vec = None if backend == "hash" else _llama_embed(piece)

        if backend == "llama" and llama_vec is None:
            conn.close()
            raise RuntimeError(
                "llama-server embeddings stopped responding while indexing."
            )

        if llama_vec is not None:
            vec = llama_vec
            used_llama = True
        else:
            vec = _hash_embed(piece)
            used_llama = False

        _lock_backend(conn, vec, used_llama)
        conn.execute(
            """
            INSERT INTO chunks (collection, source, text, vector, dim, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                collection,
                source,
                piece,
                vec.astype(np.float32).tobytes(),
                int(vec.shape[0]),
                time.time(),
            ),
        )
        added += 1
    conn.commit()
    conn.close()
    return {"added": added, "source": source, "collection": collection}


def _lock_backend(conn, vec: np.ndarray, used_llama: bool):
    if _meta_get(conn, "backend"):
        return
    _meta_set(conn, "backend", "llama" if used_llama else "hash")
    _meta_set(conn, "dim", int(vec.shape[0]))


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= CHUNK_CHARS:
            buf = f"{buf}\n{para}".strip() if buf else para
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= CHUNK_CHARS:
            buf = para
        else:
            start = 0
            while start < len(para):
                end = min(start + CHUNK_CHARS, len(para))
                piece = para[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(para):
                    buf = ""
                    break
                start = max(0, end - CHUNK_OVERLAP)
    if buf:
        chunks.append(buf)
    return chunks


def _read_text_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if ext == ".json":
        try:
            data = json.loads(raw)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return raw
    return raw


def _read_pdf(path: str) -> str:
    """Extracts text from a PDF's text layer. Scanned/image-only PDFs will
    come back empty or near-empty -- those need OCR first, which this
    function does not do."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF support needs the 'pypdf' package. Install it with: "
            "pip install pypdf"
        ) from e

    reader = PdfReader(path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            pages_text.append(f"[page {i + 1}]\n{page_text.strip()}")

    text = "\n\n".join(pages_text)
    if not text.strip():
        raise ValueError(
            "No extractable text found in this PDF -- it may be a scanned "
            "image without a text layer, which needs OCR before it can be ingested."
        )
    return text


def _read_docx(path: str) -> str:
    """Extracts paragraph and table text from a .docx file."""
    try:
        import docx
    except ImportError as e:
        raise RuntimeError(
            "Word support needs the 'python-docx' package. Install it with: "
            "pip install python-docx"
        ) from e

    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n\n".join(parts)
    if not text.strip():
        raise ValueError("No extractable text found in this .docx file.")
    return text


def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXTS:
        return _read_pdf(path)
    if ext in DOCX_EXTS:
        return _read_docx(path)
    return _read_text_file(path)


def ingest_file(path: str, collection: str) -> dict:
    collection = collection.lower()
    if collection not in VALID_COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in ALL_EXTS:
        raise ValueError(
            "Supported files: " + ", ".join(sorted(ALL_EXTS))
        )

    text = _read_file(path)
    return ingest_text(text, collection, source=os.path.basename(path))


def collection_count(collection: str) -> int:
    collection = collection.lower()
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE collection = ?",
        (collection,),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else 0


def clear_collection(collection: str) -> int:
    collection = collection.lower()
    conn = _connect()
    cur = conn.execute("DELETE FROM chunks WHERE collection = ?", (collection,))
    deleted = cur.rowcount
    leftover = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if leftover == 0:
        conn.execute("DELETE FROM meta")
    conn.commit()
    conn.close()
    return deleted


def retrieve(query: str, collection: str, k: int = TOP_K) -> list[dict]:
    collection = collection.lower()
    if not query or collection not in VALID_COLLECTIONS:
        return []

    conn = _connect()
    rows = conn.execute(
        "SELECT id, source, text, vector, dim FROM chunks WHERE collection = ?",
        (collection,),
    ).fetchall()
    conn.close()
    if not rows:
        return []

    qvec = embed_text(query)
    hits = []
    for row_id, source, text, blob, dim in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.shape[0] != qvec.shape[0]:
            continue
        score = float(np.dot(qvec, vec))
        hits.append(
            {
                "id": row_id,
                "source": source,
                "text": text,
                "score": score,
            }
        )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def format_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] source={hit['source']} relevance={hit['score']:.3f}\n{hit['text']}"
        )
    return "\n\n".join(parts)