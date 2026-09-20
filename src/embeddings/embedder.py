from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        embeddings = self.model.encode(texts)
        return np.asarray(embeddings, dtype="float32")