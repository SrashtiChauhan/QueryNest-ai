from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

similarity_matrix = cosine_similarity(embeddings)

print("\nCosine similarity matrix: ")
print(similarity_matrix)

# Reduce 384 dimensions to 2 dimensions
pca=PCA(n_components=2)
embeddings_2d=pca.fit_transform(embeddings)

print("\nFirst sentence:")
print(sentences[0])

print("\nFirst embedding:")
print(embeddings[0])

for i , sentence in enumerate(sentences):
    print(f"\nSentence {i}:")
    print(sentence)
    print("Coordinates:", embeddings_2d[i])


plt.figure(figsize=(10, 7))

plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1]
)
for i, sentence in enumerate(sentences):
    plt.annotate(
        f"Sentence {i}",
        (embeddings_2d[i, 0], embeddings_2d[i, 1])
    )

plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.title("Sentence Embeddings in 2D")
plt.grid(True)

# plt.show()
plt.savefig("semantic_embeddings.png", dpi=300, bbox_inches="tight")
print("\nVisualization saved as semantic_embeddings.png")