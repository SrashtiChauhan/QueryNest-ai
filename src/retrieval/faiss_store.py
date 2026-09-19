import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    def search(self, query_embedding, k=2):
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        return scores, indices

    @property
    def total_vectors(self):
        return self.index.ntotal
