from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document


class ChunkIndex:
    """
    O(1) lookups over resume chunks.

    Built once from the full chunk list and reused across every query, replacing
    the per-query linear scans (`next(c for c in all_chunks if ...)`) that made
    reranking and summarization O(candidates x chunks).
    """

    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        self._by_key: Dict[Tuple[Any, Any], Document] = {}
        self._by_resume: Dict[Any, List[Document]] = {}

        for c in chunks:
            rid = c.metadata.get("resume_id")
            cid = c.metadata.get("chunk_id")
            self._by_key[(rid, cid)] = c
            self._by_resume.setdefault(rid, []).append(c)

    def get(self, resume_id: Any, chunk_id: Any) -> Optional[Document]:
        """Return the chunk for a (resume_id, chunk_id) pair, or None."""
        return self._by_key.get((resume_id, chunk_id))

    def resume_chunks(self, resume_id: Any) -> List[Document]:
        """Return all chunks belonging to a resume (in original order)."""
        return self._by_resume.get(resume_id, [])

    def resume_text(self, resume_id: Any) -> str:
        """Return the full resume text, joined from its chunks."""
        return "\n\n".join(c.page_content for c in self.resume_chunks(resume_id))
