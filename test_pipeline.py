"""Test the full RAG pipeline"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import ResumeRAGPipeline
from src.config import OPENAI_API_KEY

# Initialize pipeline
print("Initializing pipeline...")
pipeline = ResumeRAGPipeline(api_key=OPENAI_API_KEY)

# Test query
query = "Find candidates with PyTorch and deep learning experience"
print(f"\nQuery: {query}")
print("Searching...\n")

# Run search
results = pipeline.search(query, top_k_summarize=5)

# Display results
print(f"Found {len(results)} top candidates:\n")
for i, result in enumerate(results, 1):
    print(f"{'='*60}")
    print(f"RANK {i} | Resume: {result['resume_id']}")
    print(f"RRF Score: {result.get('rrf_score', 'N/A'):.5f}")
    print(f"CE Score: {result.get('ce_score', 'N/A'):.4f}")
    print(f"{'='*60}")
    print(result['summary'])
    print()