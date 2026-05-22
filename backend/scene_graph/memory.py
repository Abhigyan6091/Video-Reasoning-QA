"""
Memory system – FAISS vector store + SentenceTransformers for semantic retrieval.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)

_embedder = None
_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] = []  # parallel list: index i → metadata for vector i


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedder


def _index_path() -> Path:
    return settings.FAISS_INDEX_DIR / "index.faiss"


def _meta_path() -> Path:
    return settings.FAISS_INDEX_DIR / "metadata.pkl"


def load_index() -> None:
    """Load FAISS index and metadata from disk if they exist."""
    global _index, _metadata
    settings.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    idx_p = _index_path()
    meta_p = _meta_path()

    if idx_p.exists() and meta_p.exists():
        _index = faiss.read_index(str(idx_p))
        with open(meta_p, "rb") as f:
            _metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index with {_index.ntotal} vectors")
    else:
        dim = _get_embedder().get_sentence_embedding_dimension()
        _index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim)
        _metadata = []
        logger.info("Created new FAISS index")


def save_index() -> None:
    """Persist FAISS index and metadata to disk."""
    if _index is None:
        return
    settings.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_index, str(_index_path()))
    with open(_meta_path(), "wb") as f:
        pickle.dump(_metadata, f)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode texts to normalized embeddings."""
    embedder = _get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def add_to_index(texts: list[str], metadata_list: list[dict]) -> list[int]:
    """
    Add texts to the FAISS index. Returns list of assigned IDs.
    """
    global _index, _metadata
    if _index is None:
        load_index()

    embeddings = embed_texts(texts)
    start_id = _index.ntotal
    _index.add(embeddings)  # type: ignore[union-attr]
    _metadata.extend(metadata_list)
    save_index()

    return list(range(start_id, start_id + len(texts)))


def search(query: str, top_k: int = 5, video_id: str | None = None) -> list[dict]:
    """
    Semantic search: return top-k most similar items.
    Optionally filter by video_id.
    """
    if _index is None or _index.ntotal == 0:
        return []

    q_emb = embed_texts([query])
    # Search more than needed if filtering
    search_k = min(top_k * 3, _index.ntotal)
    scores, indices = _index.search(q_emb, search_k)  # type: ignore[union-attr]

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        meta = _metadata[idx].copy()
        if video_id and meta.get("video_id") != video_id:
            continue
        meta["similarity_score"] = float(score)
        meta["embedding_id"] = int(idx)
        results.append(meta)
        if len(results) >= top_k:
            break

    return results


def compute_pairwise_similarity(texts: list[str]) -> np.ndarray:
    """
    Compute NxN cosine similarity matrix for a list of texts.
    Used for duplicate detection in the uniqueness filter.
    """
    embeddings = embed_texts(texts)
    # Normalized → cosine sim = dot product
    sim_matrix = embeddings @ embeddings.T
    return sim_matrix
