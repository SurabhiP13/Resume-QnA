"""Quick test to verify all modules load correctly"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    print("Testing imports...")
    
    try:
        from src.config import OPENAI_API_KEY, CHUNKS_PATH, EMBEDDINGS_PATH
        print("✅ Config imported")
        print(f"   Chunks path: {CHUNKS_PATH}")
        print(f"   Embeddings path: {EMBEDDINGS_PATH}")
        
        from src.generation.utils import split_resume_into_sections, smart_truncate_resume
        print("✅ Utils imported")
        
        from src.generation.summarizer import ResumeSummarizer
        print("✅ Summarizer imported")
        
        from src.pipeline import ResumeRAGPipeline
        print("✅ Pipeline imported")
        
        print("\n🎉 All imports successful!")
        
        # Test utility functions
        test_text = "## Education\nBachelor of Science\n## Experience\nSoftware Engineer"
        sections = split_resume_into_sections(test_text)
        print(f"\n✅ split_resume_into_sections works: {len(sections)} sections found")
        
        truncated = smart_truncate_resume(sections, 100)
        print(f"✅ smart_truncate_resume works: {len(truncated)} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()