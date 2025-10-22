"""
Script to run the Resume RAG pipeline
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import ResumeRAGPipeline
from src.config import CHUNKS_PATH, EMBEDDINGS_PATH, OPENAI_API_KEY

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Resume RAG Search")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--chunks", type=str, default=str(CHUNKS_PATH), help="Chunks pickle path")
    parser.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_PATH), help="Embeddings pickle path")
    args = parser.parse_args()
    
    # Initialize pipeline
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    pipeline = ResumeRAGPipeline(api_key=api_key)
    pipeline.load_data(
        chunks_path=args.chunks,
        embeddings_path=args.embeddings
    )
    
    # Run search
    results = pipeline.search(args.query, top_k_summarize=args.top_k)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"RANK {i} | Resume ID: {result['resume_id']}")
        rrf = result.get('rrf_score')
        ce = result.get('ce_score')
        print(f"RRF Score: {rrf:.5f if rrf else 'N/A'} | CE Score: {ce:.4f if ce else 'N/A'}")
        print(f"\n{result['summary']}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()