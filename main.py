from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline


load_dotenv()


document = """
Python is a high-level programming language.
It is widely used in web development and data science.
Python is also popular for automation and artificial intelligence.
Many developers use Python because its syntax is simple and readable.
"""


rag = RAGPipeline()

# Add document to the RAG system
rag.ingest(document)

# Ask a question
question = "What is Python used for?"

answer, chunks, scores = rag.query(question, k=2)

print("\nQuestion:")
print(question)

print("\nRetrieved Chunks:")

for i, (chunk, score) in enumerate(
    zip(chunks, scores[0]),
    start=1
):
    print(f"\nRank {i}")
    print("Score:", score)
    print("Chunk:", chunk)

print("\nFinal Answer:")
print(answer)

