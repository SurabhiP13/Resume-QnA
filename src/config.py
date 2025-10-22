import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks" / "resume_chunks_openai.pkl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings" / "resume_embeddings_openai.pkl"
MARKDOWN_DIR = PROCESSED_DIR / "markdown" / "Resume-markdown-docling"
# Retrieval Configuration
BM25_TOP_K = 200
DENSE_TOP_K = 200
RERANK_TOP_K = 180
RRF_K = 60
RRF_WEIGHTS = {"bm25": 2.0, "dense": 1.0}

# Generation Configuration
SUMMARY_TOP_N = 5
MAX_RESUME_CHARS = 6000
MAX_CONTEXT_CHARS = 2000