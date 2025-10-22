import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from langchain_core.documents import Document

class DenseRetriever:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.docs: List[Document] = []
        self.Xn: np.ndarray = None
        self.dim: int = None
        self.model_name: str = None
    
    def fit(self, chunks: List[Document], embeddings: np.ndarray, model_name: str):
        """Index L2-normalized embeddings"""
        if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
            raise ValueError("Embeddings must be 2D and aligned with chunks")
        
        X = np.ascontiguousarray(embeddings.astype(np.float32))
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        self.Xn = X / norms
        self.dim = self.Xn.shape[1]
        self.docs = chunks
        self.model_name = model_name
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed and normalize query"""
        resp = self.client.embeddings.create(model=self.model_name, input=[query])
        q = np.array(resp.data[0].embedding, dtype=np.float32)
        return q / (np.linalg.norm(q) + 1e-12)
    
    def search(self, query: str, top_k: int = 200) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks by cosine similarity"""
        if self.Xn is None:
            raise RuntimeError("Call fit() first")
        
        q = self._embed_query(query)
        sims = self.Xn @ q
        top_idx = np.argsort(sims)[::-1][:min(top_k, len(sims))]
        
        return [{
            "rank": rank,
            "dense_score": float(sims[i]),
            "resume_id": self.docs[i].metadata.get("resume_id"),
            "chunk_id": self.docs[i].metadata.get("chunk_id"),
            "preview": self.docs[i].page_content[:400]
        } for rank, i in enumerate(top_idx, 1)]