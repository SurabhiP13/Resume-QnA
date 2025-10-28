"""Check if data files are in the correct location"""
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CHUNKS_PATH, EMBEDDINGS_PATH, MARKDOWN_DIR

print("Checking data file locations...\n")

files_to_check = {
    "Chunks": CHUNKS_PATH,
    "Embeddings": EMBEDDINGS_PATH,
    "Markdown Directory": MARKDOWN_DIR,
}

all_good = True
for name, path in files_to_check.items():
    if path.exists():
        if path.is_file():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {name}: {path} ({size:.2f} MB)")
        else:
            num_files = len(list(path.glob("*.md")))
            print(f"✅ {name}: {path} ({num_files} markdown files)")
    else:
        print(f"❌ {name}: {path} NOT FOUND")
        all_good = False

if all_good:
    print("\n🎉 All data files are in place!")
else:
    print("\n⚠️ Some files are missing. Please check the locations above.")