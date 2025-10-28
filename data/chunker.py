import re
from pathlib import Path
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document

_SPACED_CAPS_RX = re.compile(r'(?<!\w)(?:[A-Z]\s+){2,}[A-Z](?!\w)')

def _fix_spaced_caps(s: str) -> str:
    return _SPACED_CAPS_RX.sub(lambda m: m.group(0).replace(' ', ''), s)

def _container_chunking(content: str, resume_id: str) -> List[Document]:
    content = re.sub(r'<!-- image -->', '', content)
    content = _fix_spaced_caps(content)
    containers = [
        r'about\s*me', r'summary', r'profile', r'experience', r'work\s+experience',
        r'education', r'skill[s]?', r'project[s]?', r'achievement[s]?', r'award[s]?',
        r'publication[s]?', r'competition[s]?', r'hackathon[s]?', r'certification[s]?',
    ]
    pattern = rf'(?=^#{{1,6}}\s*(?:.*\b(?:{"|".join(containers)})\b).*$)'
    parts = re.split(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
    docs: List[Document] = []
    for i, chunk in enumerate(parts):
        chunk = chunk.strip()
        if chunk and len(chunk) > 50:
            docs.append(Document(page_content=chunk, metadata={"resume_id": resume_id, "chunk_id": i}))
    return docs

def chunk_markdown_files(markdown_dir: Path) -> List[Document]:
    md_files = list(markdown_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {markdown_dir}")
    all_chunks: List[Document] = []
    for md in tqdm(md_files, desc="Chunking markdown"):
        try:
            content = md.read_text(encoding="utf-8")
            all_chunks.extend(_container_chunking(content, md.stem))
        except Exception as e:
            print(f"Failed to chunk {md.name}: {e}")
    return all_chunks