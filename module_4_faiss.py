import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sentences = [
    "Python is a programming language.",
    "I write software using Python.",
    "The recipe explains how to make pizza.",
    "JavaScript is used to build web applications."
]

model=SentenceTransformer('all-MiniLM-L6-v2')
embeddings=model.encode(sentences)

print("Original embedding shape:", embeddings.shape)

embeddings=np.asarray(embeddings, dtype="float32")
print("FAISS embedding shape:", embeddings.shape)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

print("Vector dimension:", dimension)

index=faiss.IndexFlatIP(dimension)
print("FAISS index created successfully")
index.add(embeddings)
print("Number of vectors in the index:", index.ntotal)


query = "What language is used for programming?"
query_embedding=model.encode([query])
query_embedding = np.asarray(query_embedding, dtype="float32")
faiss.normalize_L2(query_embedding)

k=2
scores, indices=index.search(query_embedding, k)
print("\nQuery:", query)

print("\nTop results:")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    print(f"\nRank {rank}")
    print("Sentence:", sentences[idx])
    print("Score:", score)
