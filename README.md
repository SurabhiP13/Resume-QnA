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
    Summarizes top candidates

## Folder Structure
rag-resume-screening/
├── data/
│   ├── raw/                # Place your PDF resumes here
│   ├── processed/          # Output folders for markdown, chunks, embeddings
│   ├── chunker.py          # Chunks markdown resumes
│   ├── loader.py           # Converts PDFs to markdown
│   ├── embed.py            # Embedding generator
│   └── __init__.py
├── scripts/
│   ├── generate_embeddings.py  # Runs the full pipeline
│   └── run_retrieval.py        # Runs a search query
├── src/
│   ├── pipeline.py         # Main pipeline logic
│   ├── config.py           # Configuration and paths
│   ├── generation/         # Summarization code
│   ├── retrieval/          # BM25, dense, fusion, reranker
│   └── __init__.py
├── tests/                  # Unit tests
├── notebooks/              # Prototyping and experiments
├── requirements.txt
└── README.md




