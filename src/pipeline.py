# import pickle
# from pathlib import Path
# from typing import List, Dict, Any

# from .data.chunker import process_resumes
# from .embeddings.generator import EmbeddingGenerator
# from .retrieval.bm25_retriever import BM25Retriever
# from .retrieval.dense_retriever import DenseRetriever
# from .retrieval.fusion import rrf_fuse
# from .retrieval.reranker import CrossEncoderReranker
# from .generation.summarizer import ResumeSummarizer

# class ResumeRAGPipeline:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.bm25 = BM25Retriever()
#         self.dense = DenseRetriever(api_key)
#         self.reranker = CrossEncoderReranker()
#         self.summarizer = ResumeSummarizer(api_key)
#         self.chunks = None
#         self.embeddings = None
    
#     def load_data(self, chunks_path: str, embeddings_path: str):
#         """Load preprocessed chunks and embeddings"""
#         with open(chunks_path, 'rb') as f:
#             self.chunks = pickle.load(f)
#         with open(embeddings_path, 'rb') as f:
#             self.embeddings = pickle.load(f)
        
#         # Fit retrievers
#         self.bm25.fit(self.chunks)
#         self.dense.fit(self.chunks, self.embeddings, "text-embedding-3-small")
    
#     def search(
#         self,
#         query: str,
#         top_k_retrieve: int = 200,
#         top_k_rerank: int = 180,
#         top_k_summarize: int = 5
#     ) -> List[Dict[str, Any]]:
#         """End-to-end RAG pipeline"""
#         # 1. Retrieve with BM25 and Dense
#         bm25_hits = self.bm25.search(query, top_k=top_k_retrieve)
#         dense_hits = self.dense.search(query, top_k=top_k_retrieve)
        
#         # 2. Fuse results
#         fused = rrf_fuse(
#             bm25_hits, dense_hits,
#             k=60, top_k=top_k_rerank,
#             weights={"bm25": 2.0, "dense": 1.0}
#         )
        
#         # 3. Rerank with cross-encoder
#         reranked = self.reranker.rerank(query, fused, self.chunks, top_k=top_k_rerank)
        
#         # 4. Generate summaries
#         summaries = self.summarizer.summarize(
#             query, reranked, self.chunks, top_n=top_k_summarize
#         )
        
#         return summaries






import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import CHUNKS_PATH, EMBEDDINGS_PATH, OPENAI_API_KEY
from .retrieval.bm25_retriever import BM25Retriever
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
        
        # Initialize components
        print("Initializing retrievers...")
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.api_key)
        self.reranker = CrossEncoderReranker()
        self.summarizer = ResumeSummarizer(self.api_key)
        
        # Fit retrievers
        print("Fitting BM25 retriever...")
        self.bm25.fit(self.chunks)
        
        print("Fitting Dense retriever...")
        self.dense.fit(self.chunks, self.embeddings, "text-embedding-3-small")
        
        print(f"✅ Pipeline initialized with {len(self.chunks)} chunks")
    
    def search(
        self,
        query: str,
        top_k_retrieve: int = 200,
        top_k_rerank: int = 180,
        top_k_summarize: int = 5
    ) -> List[Dict[str, Any]]:
        """End-to-end RAG pipeline"""
        print(f"\nSearching for: {query}")
        
        # 1. Retrieve with BM25 and Dense
        print("BM25 retrieval...")
        bm25_hits = self.bm25.search(query, top_k=top_k_retrieve)
        
        print("Dense retrieval...")
        dense_hits = self.dense.search(query, top_k=top_k_retrieve)
        
        # 2. Fuse results
        print("Fusing results...")
        fused = rrf_fuse(
            bm25_hits, dense_hits,
            k=60, top_k=top_k_rerank,
            weights={"bm25": 2.0, "dense": 1.0}
        )
        
        # 3. Rerank with cross-encoder
        print("Reranking...")
        reranked = self.reranker.rerank(query, fused, self.chunks, top_k=top_k_rerank)
        
        # 4. Generate summaries
        print("Generating summaries...")
        summaries = self.summarizer.summarize(
            query, reranked, self.chunks, top_n=top_k_summarize
        )
        
        print(f"Found {len(summaries)} candidates\n")
        return summaries