from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = Path("data/processed")


def _client_dir(client_id: str) -> Path:
    path = DATA_DIR / client_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ingest_docs(client_id: str, docs: Iterable[Dict[str, str]]) -> None:
    """Ingest a collection of documents for a client."""
    client_path = _client_dir(client_id)
    all_chunks: List[Dict[str, str]] = []
    for doc in docs:
        text = doc["text"]
        doc_id = doc["id"]
        for i, chunk in enumerate(_chunk_text(text)):
            all_chunks.append({"id": f"{doc_id}_{i}", "text": chunk})
    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": all_chunks}, client_path / "index.joblib")
    # store original docs for reference
    with open(client_path / "docs.json", "w", encoding="utf-8") as f:
        json.dump(list(docs), f)


def ingest_files(client_id: str, files: Iterable[Path]) -> None:
    docs: List[Dict[str, str]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception as exc:
                raise RuntimeError("pypdf is required for PDF ingestion") from exc
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = path.read_text(encoding="utf-8")
        docs.append({"id": path.stem, "text": text})
    ingest_docs(client_id, docs)


def query_index(client_id: str, question: str, top_k: int = 3) -> List[Dict[str, str]]:
    client_path = _client_dir(client_id)
    index_file = client_path / "index.joblib"
    if not index_file.exists():
        raise FileNotFoundError(f"No index found for client {client_id}")
    data = joblib.load(index_file)
    vectorizer: TfidfVectorizer = data["vectorizer"]
    matrix = data["matrix"]
    chunks: List[Dict[str, str]] = data["chunks"]
    q_vec = vectorizer.transform([question])
    sims = linear_kernel(q_vec, matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]
