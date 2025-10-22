import numpy as np
from openai import OpenAI
from tqdm import tqdm
from typing import List

class EmbeddingGenerator:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate OpenAI embeddings for text chunks"""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([
                np.array(item.embedding, dtype=np.float32) 
                for item in resp.data
            ])
        return np.vstack(embeddings) if embeddings else np.zeros((0, 0))