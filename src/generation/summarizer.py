from typing import List, Dict, Any
from openai import OpenAI
from langchain_core.documents import Document
from .utils import split_resume_into_sections, smart_truncate_resume

class ResumeSummarizer:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def gather_full_resumes(
        self,
        fused_results: List[Dict[str, Any]],
        all_chunks: List[Document]
    ) -> Dict[str, str]:
        """Collect all chunks for each resume_id"""
        resume_ids = {r["resume_id"] for r in fused_results if r.get("resume_id")}
        
        resume_content = {}
        for rid in resume_ids:
            chunks = [
                c.page_content for c in all_chunks
                if c.metadata.get("resume_id") == rid
            ]
            resume_content[rid] = "\n\n".join(chunks)
        
        return resume_content
    
    def summarize(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        all_chunks: List[Document],
        top_n: int = 5,
        max_resume_chars: int = 6000,
        max_context_chars: int = 2000
    ) -> List[Dict[str, Any]]:
        """Generate summaries for top N resumes"""
        top_resume_ids = []
        resume_matched_chunks = {}
        
        for r in fused_results:
            rid = r.get("resume_id")
            if rid:
                if rid not in resume_matched_chunks:
                    resume_matched_chunks[rid] = []
                resume_matched_chunks[rid].append(r)
                
                if rid not in top_resume_ids:
                    top_resume_ids.append(rid)
                
                if len(top_resume_ids) >= top_n:
                    break
        
        resume_content = self.gather_full_resumes(fused_results, all_chunks)
        
        summaries = []
        for rid in top_resume_ids:
            full_resume = resume_content.get(rid, "")
            matched_chunks = resume_matched_chunks.get(rid, [])
            
            matched_texts = []
            for m in matched_chunks[:3]:
                chunk = next((
                    c for c in all_chunks
                    if c.metadata.get("resume_id") == rid
                    and c.metadata.get("chunk_id") == m.get("chunk_id")
                ), None)
                if chunk:
                    matched_texts.append(chunk.page_content)
            
            matched_context = "\n---\n".join(matched_texts) if matched_texts else "N/A"
            
            resume_sections = split_resume_into_sections(full_resume)
            truncated_resume = smart_truncate_resume(resume_sections, max_resume_chars)
            
            prompt = f"""You are a recruiter assistant. Analyze this resume and provide:
1. A brief summary of the candidate's profile (2-3 sentences)
2. How this candidate matches the query: "{query}"
3. Key strengths relevant to the query

Resume ID: {rid}

MOST RELEVANT SECTIONS (from search):
{matched_context[:max_context_chars]}

FULL RESUME:
{truncated_resume}

Focus on the relevant sections above, but use the full resume for complete context.
Provide a concise response.
Also, if there are any duplicate resumes, make sure to output only one summary per unique resume."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            summaries.append({
                "resume_id": rid,
                "summary": response.choices[0].message.content,
                "rrf_score": next((r["rrf_score"] for r in fused_results if r.get("resume_id") == rid), None),
                "ce_score": next((r.get("ce_score") for r in fused_results if r.get("resume_id") == rid), None),
                "matched_sections": len(matched_chunks)
            })
        
        return summaries