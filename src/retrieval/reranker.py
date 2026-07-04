import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from .chunk_index import ChunkIndex

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        chunk_index: ChunkIndex,
        top_k: int = 180
    ) -> List[Dict[str, Any]]:
        """Rerank fused results using cross-encoder"""
        pairs = []
        valid_results = []

        for r in fused_results:
            chunk = chunk_index.get(r.get("resume_id"), r.get("chunk_id"))

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