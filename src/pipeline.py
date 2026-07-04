import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import (
    BM25_CACHE_PATH,
    BM25_TOP_K,
    CHUNKS_PATH,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    LLM_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_RESUME_CHARS,
    OPENAI_API_KEY,
    RERANK_TOP_K,
    RERANKER_MODEL,
    RRF_K,
    RRF_WEIGHTS,
    SUMMARY_TOP_N,
)
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.chunk_index import ChunkIndex
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.fusion import rrf_fuse
from .retrieval.reranker import CrossEncoderReranker
from .generation.summarizer import ResumeSummarizer

class ResumeRAGPipeline:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        chunks_path: Optional[str] = None,
        embeddings_path: Optional[str] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
            chunks_path: Path to chunks pickle (uses config if not provided)
            embeddings_path: Path to embeddings pickle (uses config if not provided)
        """
        # Use provided values or fall back to config
        self.api_key = api_key or OPENAI_API_KEY
        chunks_path = chunks_path or CHUNKS_PATH
        embeddings_path = embeddings_path or EMBEDDINGS_PATH
        
        # Load data
        print(f"Loading chunks from: {chunks_path}")
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"Loading embeddings from: {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)

        # Build a single lookup index over chunks, reused by rerank/summarize
        # instead of scanning the full chunk list on every query.
        self.chunk_index = ChunkIndex(self.chunks)

        # Initialize components
        print("Initializing retrievers...")
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.api_key)
        self.reranker = CrossEncoderReranker(RERANKER_MODEL)
        self.summarizer = ResumeSummarizer(self.api_key, model=LLM_MODEL)

        # Fit retrievers. BM25 fitting is cached to disk keyed on the chunks
        # file's stat, so it is only recomputed when the chunks actually change.
        print("Fitting BM25 retriever...")
        self.bm25.fit(
            self.chunks,
            cache_path=str(BM25_CACHE_PATH),
            cache_key=self._bm25_cache_key(chunks_path),
        )

        print("Fitting Dense retriever...")
        self.dense.fit(self.chunks, self.embeddings, EMBEDDING_MODEL)

        print(f"✅ Pipeline initialized with {len(self.chunks)} chunks")

    @staticmethod
    def _bm25_cache_key(chunks_path: str) -> Optional[str]:
        """Cache key derived from the chunks file identity (path, mtime, size)."""
        try:
            st = os.stat(chunks_path)
        except OSError:
            return None
        return f"{chunks_path}:{st.st_mtime_ns}:{st.st_size}"
    
    def search(
        self,
        query: str,
        bm25_top_k: int = BM25_TOP_K,
        dense_top_k: int = DENSE_TOP_K,
        top_k_rerank: int = RERANK_TOP_K,
        top_k_summarize: int = SUMMARY_TOP_N,
    ) -> List[Dict[str, Any]]:
        """End-to-end RAG pipeline"""
        print(f"\nSearching for: {query}")

        # 1. Retrieve with BM25 and Dense
        print("BM25 retrieval...")
        bm25_hits = self.bm25.search(query, top_k=bm25_top_k)

        print("Dense retrieval...")
        dense_hits = self.dense.search(query, top_k=dense_top_k)

        # 2. Fuse results
        print("Fusing results...")
        fused = rrf_fuse(
            bm25_hits, dense_hits,
            k=RRF_K, top_k=top_k_rerank,
            weights=RRF_WEIGHTS
        )

        # 3. Rerank with cross-encoder
        print("Reranking...")
        reranked = self.reranker.rerank(query, fused, self.chunk_index, top_k=top_k_rerank)

        # 4. Generate summaries
        print("Generating summaries...")
        summaries = self.summarizer.summarize(
            query, reranked, self.chunk_index,
            top_n=top_k_summarize,
            max_resume_chars=MAX_RESUME_CHARS,
            max_context_chars=MAX_CONTEXT_CHARS,
        )

        print(f"Found {len(summaries)} candidates\n")
        return summaries