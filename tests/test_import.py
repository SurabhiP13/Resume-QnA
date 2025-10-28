import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
def test_imports():
    try:
        from data.loader import convert_pdfs_to_markdown
        from data.chunker import chunk_markdown_files
        from data.embed import EmbeddingGenerator
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.fusion import rrf_fuse
        from src.retrieval.reranker import CrossEncoderReranker
        from src.generation.summarizer import ResumeSummarizer

    except Exception as e:
        assert False, f"Import failed: {e}"