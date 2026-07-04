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

def print_results(query, results):
    """Pretty-print search results for a single query."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        print(f"RANK {i} | Resume ID: {result['resume_id']}")
        rrf = result.get('rrf_score')
        ce = result.get('ce_score')
        rrf_str = f"{rrf:.5f}" if rrf is not None else "N/A"
        ce_str = f"{ce:.4f}" if ce is not None else "N/A"
        print(f"RRF Score: {rrf_str} | CE Score: {ce_str}")
        print(f"\n{result['summary']}\n")
        print("-" * 60)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Resume RAG Search")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Reuse one loaded pipeline to answer many queries (avoids cold start per query)",
    )
    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("provide --query or run with --interactive")

    # Initialize pipeline
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Loading pickles, fitting BM25/dense, and loading the cross-encoder happens
    # once here; interactive mode amortizes it across every query in the session.
    pipeline = ResumeRAGPipeline(api_key=api_key)

    if args.interactive:
        print("\nInteractive mode. Enter a query (blank line or Ctrl-D to exit).")
        while True:
            try:
                query = input("\nquery> ").strip()
            except EOFError:
                break
            if not query:
                break
            results = pipeline.search(query, top_k_summarize=args.top_k)
            print_results(query, results)
    else:
        results = pipeline.search(args.query, top_k_summarize=args.top_k)
        print_results(args.query, results)


if __name__ == "__main__":
    main()