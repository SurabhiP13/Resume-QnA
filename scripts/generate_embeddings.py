"""
Complete pipeline: PDF → Markdown → Chunks → Embeddings
Run this to process all resumes from data/raw/
"""
import sys
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import convert_pdfs_to_markdown
from data.chunker import chunk_markdown_files
from data.embed import EmbeddingGenerator

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

# Paths
PDF_DIR = project_root / "data" / "raw"
MARKDOWN_DIR = project_root / "data" / "processed" / "markdown" / "Resume-markdown-docling"
CHUNKS_DIR = project_root / "data" / "processed" / "chunks"
EMBEDDINGS_DIR = project_root / "data" / "processed" / "embeddings"

CHUNKS_PATH = CHUNKS_DIR / "resume_chunks_openai.pkl"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "resume_embeddings_openai.pkl"

# Create directories
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def main(max_resumes: int = 200):
    print("STEP 1: Converting PDFs to Markdown")
    converted = convert_pdfs_to_markdown(PDF_DIR, MARKDOWN_DIR, max_resumes=max_resumes)
    if converted == 0:
        raise RuntimeError("No PDFs converted")
    print(f"Converted {converted} PDFs\n")

    print("STEP 2: Chunking Markdown Files")
    chunks = chunk_markdown_files(MARKDOWN_DIR)
    if not chunks:
        raise RuntimeError("No chunks produced")
    print(f"Created {len(chunks)} chunks\n")

    print("STEP 3: Generating Embeddings")
    texts = [d.page_content for d in chunks]
    gen = EmbeddingGenerator(api_key=API_KEY)
    embeddings = gen.generate(texts, model="text-embedding-3-small", batch_size=100)
    print(f"Generated embeddings: {embeddings.shape}\n")

    print("STEP 4: Saving to Disk")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks -> {CHUNKS_PATH}")
    
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings -> {EMBEDDINGS_PATH}")
    
    print("\nDone. Pipeline ready to use.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-resumes", type=int, default=200, help="Max PDFs to process")
    args = p.parse_args()
    main(max_resumes=args.max_resumes)