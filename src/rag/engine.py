# src/rag/engine.py
"""
Production RAG engine – FAISS vector search backed by sentence-transformers.

Build phase  (run on laptop via scripts/build_rag_index.py):
    Text files → chunks → embeddings → faiss.index + documents.pkl

Runtime phase (on Pi):
    Query → embed → FAISS search → top-k document chunks returned
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import faiss
    _FAISS_OK = True
except ImportError:
    logger.warning("faiss-cpu not installed – RAG disabled")
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    logger.warning("sentence-transformers not installed – RAG disabled")
    _ST_OK = False


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    On initialisation it loads a pre-built FAISS index and the corresponding
    document chunks from disk.  At query time it embeds the query with the
    same model used at index build time, searches the index, and returns the
    top-k chunks as dictionaries.
    """

    INDEX_FILE     = "faiss.index"
    DOCUMENTS_FILE = "documents.pkl"

    def __init__(self, config: dict):
        self.index_path      = config.get("index_path", "data/index")
        self.embedding_model = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
        self.top_k           = config.get("top_k", 3)

        self._index:     Optional[Any]          = None   # faiss.Index
        self._documents: List[Dict[str, Any]]   = []
        self._encoder:   Optional[Any]          = None   # SentenceTransformer

        self._load()

    # ── initialisation helpers ────────────────────────────────────────────────

    def _load(self):
        """Load FAISS index, documents, and embedding model."""
        if not (_FAISS_OK and _ST_OK):
            logger.warning("RAG dependencies missing – running in keyword-fallback mode")
            return

        index_file = os.path.join(self.index_path, self.INDEX_FILE)
        docs_file  = os.path.join(self.index_path, self.DOCUMENTS_FILE)

        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            logger.warning(
                "RAG index not found at '%s'. "
                "Run scripts/build_rag_index.py on your laptop first, "
                "then copy data/index/ to the Pi.",
                self.index_path,
            )
            return

        try:
            logger.info("Loading FAISS index from %s", index_file)
            self._index = faiss.read_index(index_file)

            with open(docs_file, "rb") as f:
                self._documents = pickle.load(f)

            logger.info(
                "Loaded %d document chunks (index size: %d vectors)",
                len(self._documents),
                self._index.ntotal,
            )
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
            self._index     = None
            self._documents = []
            return

        # Load embedding model (required on Pi for query vectorisation)
        try:
            logger.info("Loading embedding model: %s", self.embedding_model)
            self._encoder = SentenceTransformer(self.embedding_model)
            logger.info("✅ RAG engine ready (FAISS + sentence-transformers)")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            self._encoder = None

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        tags: List[str] = None,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for *query*.

        Args:
            query:  Natural-language search string.
            tags:   Optional list of tag strings to pre-filter candidates.
                    When provided, only chunks whose tag list overlaps are
                    considered before the vector search.
            top_k:  Override the default number of results.

        Returns:
            List of document dicts, each containing at least:
                - ``content``  (str)  – the raw text chunk
                - ``source``   (str)  – originating filename
                - ``tags``     (list) – metadata tags
                - ``score``    (float)– L2 distance (lower = more similar)
        """
        if top_k is None:
            top_k = self.top_k

        if not self._index or not self._encoder or not self._documents:
            return self._keyword_fallback(query, tags, top_k)

        try:
            # 1. Encode query
            query_vec = self._encoder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)  # shape (1, D)

            # 2. Tag pre-filter: build a subset index if tags given
            if tags:
                candidates = [
                    (i, doc) for i, doc in enumerate(self._documents)
                    if any(t in doc.get("tags", []) for t in tags)
                ]
                if candidates:
                    return self._search_subset(query_vec, candidates, top_k)
                # Fall through to full search if no tag matches

            # 3. Full FAISS search
            distances, indices = self._index.search(query_vec, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:          # FAISS pads with -1 when fewer results exist
                    continue
                doc = dict(self._documents[idx])
                doc["score"] = float(dist)
                results.append(doc)

            logger.debug("RAG retrieved %d chunks for query: '%s'", len(results), query[:60])
            return results

        except Exception as e:
            logger.error("RAG retrieval error: %s", e)
            return self._keyword_fallback(query, tags, top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics for diagnostics."""
        return {
            "total_documents":    len(self._documents),
            "index_vectors":      self._index.ntotal if self._index else 0,
            "embedding_model":    self.embedding_model,
            "faiss_available":    _FAISS_OK,
            "encoder_available":  self._encoder is not None,
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _search_subset(
        self,
        query_vec: "np.ndarray",
        candidates: List[tuple],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search within a tag-filtered subset by brute-force cosine similarity.
        Used when the caller provides tag filters.
        """
        indices, docs = zip(*candidates)
        embeddings = np.array(
            [self._documents[i]["embedding"] for i in indices],
            dtype=np.float32,
        )
        # Cosine similarity (vectors are already normalised)
        scores = (embeddings @ query_vec.T).flatten()
        top_idxs = np.argsort(scores)[::-1][:top_k]

        results = []
        for pos in top_idxs:
            doc = dict(docs[pos])
            doc.pop("embedding", None)  # don't return raw vector to caller
            doc["score"] = float(scores[pos])
            results.append(doc)
        return results

    def _keyword_fallback(
        self,
        query: str,
        tags: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Simple term-overlap fallback when FAISS/encoder are unavailable.
        Returns documents whose content overlaps with query tokens.
        """
        if not self._documents:
            return []

        query_tokens = set(query.lower().split())
        scored = []
        for doc in self._documents:
            # tag filter
            if tags and not any(t in doc.get("tags", []) for t in tags):
                continue
            content_tokens = set(doc.get("content", "").lower().split())
            overlap = len(query_tokens & content_tokens)
            if overlap > 0:
                scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

