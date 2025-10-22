import re
from typing import List, Dict, Any
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
    
    def fit(self, chunks: List[Document]):
        """Index document chunks"""
        self.docs = chunks
        self.doc_tokens = [
            self.clean_and_tokenize(d.page_content) for d in chunks
        ]
        self.bm25 = BM25Okapi(self.doc_tokens)
    
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