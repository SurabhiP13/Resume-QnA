from pathlib import Path
from typing import List
import re
from langchain_core.documents import Document

def fix_spaced_caps(s: str) -> str:
    """Join capital letters separated by spaces (e.g., 'S K I L L' -> 'SKILL')"""
    pattern = re.compile(r'(?<!\w)(?:[A-Z]\s+){2,}[A-Z](?!\w)')
    return pattern.sub(lambda m: m.group(0).replace(' ', ''), s)

def container_chunking(content: str, resume_id: str) -> List[Document]:
    """Split resume by semantic sections (Education, Experience, etc.)"""
    content = re.sub(r'<!-- image -->', '', content)
    content = fix_spaced_caps(content)
    
    containers = [
        r'about\s*me', r'summary', r'profile', r'experience', r'work\s+experience',
        r'education', r'skill[s]?', r'project[s]?', r'achievement[s]?', r'award[s]?',
        r'publication[s]?', r'competition[s]?', r'hackathon[s]?', r'certification[s]?',
    ]
    container_alt = '|'.join(containers)
    pattern = rf'(?=^#{{1,6}}\s*(?:.*\b(?:{container_alt})\b).*$)'
    chunks = re.split(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    
    docs = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if chunk and len(chunk) > 50:
            docs.append(Document(
                page_content=chunk,
                metadata={'resume_id': resume_id, 'chunk_id': i}
            ))
    return docs

def process_resumes(markdown_dir: Path) -> List[Document]:
    """Process all markdown resumes into chunks"""
    all_chunks = []
    for md_file in markdown_dir.glob("*.md"):
        content = md_file.read_text(encoding='utf-8')
        resume_id = md_file.stem
        chunks = container_chunking(content, resume_id)
        all_chunks.extend(chunks)
    return all_chunks