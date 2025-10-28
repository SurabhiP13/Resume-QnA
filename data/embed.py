import numpy as np
from typing import List
from tqdm import tqdm
from openai import OpenAI

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(self, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI embeddings"):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=model, input=batch)
            vectors.extend([np.array(it.embedding, dtype=np.float32) for it in resp.data])
        return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)

    def generate_query_embedding(self, query: str, model: str = "text-embedding-3-small") -> np.ndarray:
        resp = self.client.embeddings.create(model=model, input=[query])
        return np.array(resp.data[0].embedding, dtype=np.float32)