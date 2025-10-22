import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        all_chunks: List[Document],
        top_k: int = 180
    ) -> List[Dict[str, Any]]:
        """Rerank fused results using cross-encoder"""
        pairs = []
        valid_results = []
        
        for r in fused_results:
            chunk = next((
                c for c in all_chunks
                if c.metadata.get("resume_id") == r.get("resume_id")
                and c.metadata.get("chunk_id") == r.get("chunk_id")
            ), None)
            
            if chunk:
                pairs.append((query, chunk.page_content))
                valid_results.append(r)
        
        ce_scores = self.model.predict(pairs).tolist()
        
        for result, score in zip(valid_results, ce_scores):
            result["ce_score"] = float(score)
        
        reranked = sorted(valid_results, key=lambda x: x["ce_score"], reverse=True)[:top_k]
        
        for i, r in enumerate(reranked, 1):
            r["rerank_position"] = i
        
        return reranked