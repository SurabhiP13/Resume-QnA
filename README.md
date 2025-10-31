# rag-resume-screening

Resume RAG Screening
A pipeline for automated resume screening using Retrieval-Augmented Generation (RAG).
It processes PDF resumes, chunks and embeds them, and enables semantic search and ranking.

## Features
    Converts PDF resumes to markdown using Docling
    Chunks resumes into logical sections
    Generates dense embeddings with OpenAI
    Supports keyword (BM25) and semantic (dense) retrieval
    Combines results with Reciprocal Rank Fusion (RRF)
    Reranks using Cross Encoder
    Summarizes top candidates


## Quickstart

1. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

2. **Add your OpenAI API key to `.env`**
    ```
    OPENAI_API_KEY=sk-xxxxxxx
    ```

3. **Place PDF resumes in `data/raw/`**

4. **Run the embedding pipeline**
    ```
    python scripts/generate_embeddings.py
    ```

5. **Run a search query**
    ```
    python scripts/run_retrieval.py --query "docker kubernetes" --top-k 5
    ```

## Testing

Run unit tests with:
pytest tests/

## Customization

- Edit `data/chunker.py` to change chunking logic.
- Edit `data/loader.py` to change PDF-to-markdown conversion.
- Edit `data/embed.py` for embedding model changes.
- Edit `src/pipeline.py` for retrieval and ranking logic.

## License

MIT
