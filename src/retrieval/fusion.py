import hashlib
from typing import List, Dict, Any, Optional

def rrf_fuse(
    bm25_hits: Optional[List[Dict[str, Any]]],
    dense_hits: Optional[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 180,
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion (RRF): score = Σ w_s / (k + rank_s)"""
    weights = weights or {"bm25": 1.0, "dense": 1.0}
    pool: Dict[str, Dict[str, Any]] = {}
    
    def make_key(hit: Dict[str, Any]) -> str:
        rid = str(hit.get("resume_id", "")).strip()
        cid = str(hit.get("chunk_id", "")).strip()
        if rid or cid:
            return f"{rid}::{cid}"
        prev = (hit.get("preview") or "")[:256]
        return "hash::" + hashlib.md5(prev.encode()).hexdigest()
    
    def add_source(hits: Optional[List], label: str):
        if not hits:
            return
        for h in hits:
            key = make_key(h)
            rec = pool.setdefault(key, {
                "resume_id": h.get("resume_id"),
                "chunk_id": h.get("chunk_id"),
                "preview": h.get("preview"),
                "bm25_rank": None, "bm25_score": None,
                "dense_rank": None, "dense_score": None,
                "rrf_score": 0.0
            })
            r = h.get("rank")
            if isinstance(r, int) and r >= 1:
                rec["rrf_score"] += weights[label] * (1.0 / (k + r))
            
            rank_key = f"{label}_rank"
            score_key = f"{label}_score"
            if rec[rank_key] is None:
                rec[rank_key] = r
            if rec[score_key] is None:
                val = h.get(score_key) or h.get("bm25_score") or h.get("dense_score")
                rec[score_key] = float(val) if val else None
    
    add_source(bm25_hits, "bm25")
    add_source(dense_hits, "dense")
    
    fused = sorted(pool.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
    for i, rec in enumerate(fused, 1):
        rec["rank"] = i
    return fused