from sentence_transformers import SentenceTransformer

sentences=[
    "Python is a programming language.",
    "I write software using Python.",
    "The recipe explains how to make pizza.",
    "JavaScript is used to build web applications."
]

model=SentenceTransformer('all-MiniLM-L6-v2')
embeddings=model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst sentence:")
print(sentences[0])

print("\nFirst embedding:")
print(embeddings[0])