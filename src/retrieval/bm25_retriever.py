import os
import pickle
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.docs: List[Document] = []
        self.doc_tokens: List[List[str]] = []

    @staticmethod
    def clean_and_tokenize(text: str) -> List[str]:
        """Tokenize text for BM25"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return [t for t in text.split() if t]

    def fit(
        self,
        chunks: List[Document],
        cache_path: Optional[str] = None,
        cache_key: Optional[str] = None,
    ):
        """
        Index document chunks.

        Fitting BM25 means tokenizing every chunk and building the BM25Okapi
        index, which is wasted work on every cold process start. When a
        ``cache_path`` is given, the fitted index is persisted and reused across
        process starts as long as ``cache_key`` (typically derived from the
        chunks file's stat) still matches.
        """
        self.docs = chunks

        if cache_path and self._load_cache(cache_path, cache_key, len(chunks)):
            return

        self.doc_tokens = [
            self.clean_and_tokenize(d.page_content) for d in chunks
        ]
        self.bm25 = BM25Okapi(self.doc_tokens)

        if cache_path:
            self._save_cache(cache_path, cache_key)

    def _load_cache(
        self, cache_path: str, cache_key: Optional[str], n_docs: int
    ) -> bool:
        """Load a cached BM25 index if it is present and still valid."""
        if not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except Exception:
            return False

        if cached.get("cache_key") != cache_key:
            return False
        if len(cached.get("doc_tokens", [])) != n_docs:
            return False

        self.doc_tokens = cached["doc_tokens"]
        self.bm25 = cached["bm25"]
        return True

    def _save_cache(self, cache_path: str, cache_key: Optional[str]):
        """Persist the fitted BM25 index for reuse on the next process start."""
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{cache_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {
                    "cache_key": cache_key,
                    "doc_tokens": self.doc_tokens,
                    "bm25": self.bm25,
                },
                f,
            )
        os.replace(tmp_path, cache_path)
    
    def search(self, query: str, top_k: int = 200) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks by BM25 score"""
        if not self.bm25:
            raise RuntimeError("Call fit() before search()")
        
        q_tokens = self.clean_and_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [{
            "rank": rank,
            "bm25_score": float(scores[i]),
            "resume_id": self.docs[i].metadata.get("resume_id"),
            "chunk_id": self.docs[i].metadata.get("chunk_id"),
            "preview": self.docs[i].page_content[:400]
        } for rank, i in enumerate(top_idx, 1)]