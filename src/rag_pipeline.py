import numpy as np

from src.ingestion.chunker import chunk_document
from src.embeddings.embedder import Embedder
from src.retrieval.faiss_store import FAISSStore
from src.generation.llm import LLM


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.llm = LLM()

        self.chunks = []
        self.store = None

    def ingest(self, document):
        # 1. Chunk document
        self.chunks = chunk_document(document)

        # 2. Create embeddings
        embeddings = self.embedder.encode(self.chunks)

        # 3. Create FAISS store
        dimension = embeddings.shape[1]
        self.store = FAISSStore(dimension)

        # 4. Add embeddings to FAISS
        self.store.add(embeddings)

    def query(self, question, k=2):
        # 1. Embed the question
        query_embedding = self.embedder.encode([question])

        # 2. Search FAISS
        scores, indices = self.store.search(query_embedding, k)

        # 3. Retrieve chunks
        retrieved_chunks = [
            self.chunks[idx]
            for idx in indices[0]
        ]

        # 4. Build context
        context = "\n\n".join(retrieved_chunks)

        # 5. Build prompt
        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        # 6. Generate answer
        answer = self.llm.generate(prompt)

        return answer, retrieved_chunks, scores
