# import numpy as np

# from src.ingestion.chunker import chunk_document
# from src.embeddings.embedder import Embedder
# from src.retrieval.faiss_store import FAISSStore
# from src.generation.llm import LLM


# class RAGPipeline:
#     def __init__(self):
#         self.embedder = Embedder()
#         self.llm = LLM()

#         self.chunks = []
#         self.store = None

#     def ingest(self, document):
#         # 1. Chunk document
#         self.chunks = chunk_document(document)

#         # 2. Create embeddings
#         embeddings = self.embedder.encode(self.chunks)

#         # 3. Create FAISS store
#         dimension = embeddings.shape[1]
#         self.store = FAISSStore(dimension)

#         # 4. Add embeddings to FAISS
#         self.store.add(embeddings)

#     def query(self, question, k=2):
#         # 1. Embed the question
#         query_embedding = self.embedder.encode([question])

#         # 2. Search FAISS
#         scores, indices = self.store.search(query_embedding, k)

#         # 3. Retrieve chunks
#         # retrieved_chunks = [
#         #     self.chunks[idx]
#         #     for idx in indices[0]
#         # ]
#         #check teh best imilarity score
#         best_score=scores[0][0]
#         if best_score<0.5:
#             return ("I don't have enough information in the provided context.", [], scores)
#         #recieve chunks when sc is high enough
#         retrieved_chunks = [
#             self.chunks[idx]
#             for idx in indices[0]
#         ]


#         # 4. Build context
#         context = "\n\n".join(retrieved_chunks)

#         # 5. Build prompt
#         prompt = f"""
# You are a knowledge assistant.

# Answer the question using ONLY the information provided in the context.

# Rules:
# - Do not use outside knowledge.
# - Do not make up information.
# - If the answer is not contained in the context, say:
#   "I don't have enough information in the provided context."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#         # 6. Generate answer
#         answer = self.llm.generate(prompt)

#         return answer, retrieved_chunks, scores


from src.ingestion.chunker import chunk_document
from src.embeddings.embedder import Embedder
from src.retrieval.faiss_store import FAISSStore
from src.generation.llm import LLM
# import numpy as np

class RetrievalResult:
    def __init__(self, chunk, source, score):
        self.chunk = chunk
        self.source = source
        self.score = score

    def __repr__(self):
        return (
            f"RetrievalResult("
            f"source='{self.source}', "
            f"score={self.score:.4f})"
        )
class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.llm = LLM()

        self.chunks = []
        self.sources = []
        self.store = None

    def ingest(self, documents):
        """
        Ingest multiple documents.

        Each document should contain:
        {
            "source": "file path",
            "text": "document content"
        }
        """

        self.chunks = []
        self.sources = []

        # Create chunks from every document
        for document in documents:
            document_chunks = chunk_document(document["text"])

            self.chunks.extend(document_chunks)

            # Keep the source for every chunk
            self.sources.extend(
                [document["source"]] * len(document_chunks)
            )

        # Create embeddings
        embeddings = self.embedder.encode(self.chunks)

        # Create FAISS store
        dimension = embeddings.shape[1]
        self.store = FAISSStore(dimension)

        # Add embeddings
        self.store.add(embeddings)

    def query(self, question, k=2):
        # Embed question
        query_embedding = self.embedder.encode([question])

        # Search FAISS
        scores, indices = self.store.search(
            query_embedding,
            k
        )

        # Check best similarity score
        best_score = scores[0][0]

        if best_score < 0.5:
            return (
                "I don't have enough information in the provided context.",
                []
            )

        # Retrieve chunks and their sources
        retrieved_results = []

        for idx, score in zip(indices[0], scores[0]):
            result = RetrievalResult(
                chunk=self.chunks[idx],
                source=self.sources[idx],
                score=score
            )

            retrieved_results.append(result)

        # Build context
        context = "\n\n".join(
            result.chunk
            for result in retrieved_results
)

        # Build grounded prompt
        prompt = f"""
You are a knowledge assistant.

Answer the question using ONLY the information provided in the context.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- If the answer is not contained in the context, say:
  "I don't have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        answer = self.llm.generate(prompt)

        return answer, retrieved_results