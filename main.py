from dotenv import load_dotenv

from src.ingestion.loader import load_text_files
from src.rag_pipeline import RAGPipeline


load_dotenv()

documents = load_text_files("data/raw")
rag = RAGPipeline()
rag.ingest(documents)


question = input("\nEnter your question: ")

answer, retrieved_results = rag.query(
    question,
    k=2
)


print("\nQuestion:")
print(question)


if retrieved_results:
    print("\nRetrieved Chunks:")

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):
        print(f"\nRank {i}")
        print("Score:", result.score)
        print("Source:", result.source)
        print("Chunk:", result.chunk)

else:
    print("\nNo relevant context found.")


print("\nFinal Answer:")
print(answer)