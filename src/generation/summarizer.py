from typing import List, Dict, Any
from openai import OpenAI

from ..retrieval.chunk_index import ChunkIndex
from .utils import split_resume_into_sections, smart_truncate_resume

class ResumeSummarizer:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def summarize(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        chunk_index: ChunkIndex,
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

        summaries = []
        for rid in top_resume_ids:
            full_resume = chunk_index.resume_text(rid)
            matched_chunks = resume_matched_chunks.get(rid, [])

            matched_texts = []
            for m in matched_chunks[:3]:
                chunk = chunk_index.get(rid, m.get("chunk_id"))
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
Following are some of the rules to follow while generating the summary:
1. Be objective and focus on the candidate's qualifications.
2. Avoid personal opinions or subjective statements.
3. If there are any duplicate resumes, make sure to output only one summary per unique resume.
4. If the resume lacks relevant information, state that clearly in your summary.
5. If the resume is not relevant to the query, do not generate a summary and state that there were not enough relevant resumes
If """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            top_hit = matched_chunks[0] if matched_chunks else {}
            summaries.append({
                "resume_id": rid,
                "summary": response.choices[0].message.content,
                "rrf_score": top_hit.get("rrf_score"),
                "ce_score": top_hit.get("ce_score"),
                "matched_sections": len(matched_chunks)
            })
        
        return summaries