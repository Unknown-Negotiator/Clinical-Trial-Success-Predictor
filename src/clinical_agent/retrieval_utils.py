"""
Lightweight RAG retrieval using existing FAISS index if available,
with a fallback lexical scorer when FAISS or the embedding model is absent.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class Retriever:
    def __init__(self, rag_dir: Path, use_faiss: bool = True) -> None:
        self.rag_dir = rag_dir
        self.meta = self._load_meta(rag_dir / "rag_meta.jsonl")
        self.index = None
        self.embed_model = None
        self.use_faiss = False
        if use_faiss and faiss is not None and SentenceTransformer is not None:
            try:
                self.index = faiss.read_index(str(rag_dir / "rag.faiss"))
                self.embed_model = SentenceTransformer("intfloat/e5-base")
                self.use_faiss = True
            except Exception:
                self.index = None
                self.embed_model = None
                self.use_faiss = False
        # Precompute tokens for fallback lexical scoring
        self._token_cache = [self._tokenize(m["text"]) for m in self.meta]

    def _load_meta(self, path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    @staticmethod
    def _tokenize(text: str) -> Dict[str, int]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        freqs: Dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        return freqs

    def _lexical_score(self, query: str, idx: int) -> float:
        q_tokens = self._tokenize(query)
        doc_tokens = self._token_cache[idx]
        if not q_tokens or not doc_tokens:
            return 0.0
        score = 0.0
        for tok, q_tf in q_tokens.items():
            d_tf = doc_tokens.get(tok, 0)
            if d_tf:
                score += math.log1p(q_tf) * math.log1p(d_tf)
        return score

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        if not query.strip():
            return []
        if self.use_faiss and self.index is not None and self.embed_model is not None:
            q_emb = self.embed_model.encode(f"query: {query}", normalize_embeddings=True)
            scores, idxs = self.index.search(q_emb.reshape(1, -1), k)
            hits: List[Dict[str, str]] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0 or idx >= len(self.meta):
                    continue
                meta = self.meta[idx]
                hits.append(self._format_hit(meta, float(score)))
            return hits
        # Fallback lexical
        scored = []
        for i, meta in enumerate(self.meta):
            s = self._lexical_score(query, i)
            if s > 0:
                scored.append((s, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._format_hit(meta, float(score)) for score, meta in scored[:k]]

    @staticmethod
    def _format_hit(meta: Dict[str, str], score: float) -> Dict[str, str]:
        snippet = meta.get("text", "")
        snippet = snippet.strip().replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        return {
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "source": meta.get("source", ""),
            "text": snippet,
            "score": f"{score:.3f}",
        }
