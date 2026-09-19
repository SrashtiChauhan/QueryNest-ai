from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")
print("HF token loaded:", HF_TOKEN is not None)
client = InferenceClient(token=HF_TOKEN)




document = """
Python is a high-level programming language.
It is widely used in web development and data science.
Python is also popular for automation and artificial intelligence.
Many developers use Python because its syntax is simple and readable.
"""

# chunk_size=2
# words=document.split()
# chunks=[]
# for i in range(0, len(words), chunk_size):
#     chunk = " ".join(words[i:i + chunk_size])
#     chunks.append(chunk)



chunk_size = 100
# chunk_overlap=20
#split document into sentences
# sentences=[
#     sentence.strip()
#     for sentence in document.split(".")
#     if sentence.strip()
# ]
# chunks=[]
# curr_chunk=""
# for sentence in sentences:
#     sentence=sentence + "."

#     if len(curr_chunk)+len(sentence)<=chunk_size:
#         curr_chunk+=sentence+ ""
#     else:
#         chunks.append(curr_chunk.strip())
        # Keep some overlap from the previous chunk
        # overlap = curr_chunk[-chunk_overlap:]

        # curr_chunk = overlap + " " + sentence + " "



# Split document into sentences
sentences = [
    sentence.strip()
    for sentence in document.split(".")
    if sentence.strip()
]

chunks = []
curr_chunk = ""

for sentence in sentences:
    sentence = sentence + "."

    # Add sentence if it fits
    if len(curr_chunk) + len(sentence) <= chunk_size:
        curr_chunk += sentence + " "

    else:
        # Save current chunk
        chunks.append(curr_chunk.strip())

        # Keep the last complete sentence as overlap
        previous_sentences = curr_chunk.strip().split(".")
        previous_sentences = [
            s.strip()
            for s in previous_sentences
            if s.strip()
        ]

        if previous_sentences:
            overlap = previous_sentences[-1] + "."
        else:
            overlap = ""

        curr_chunk = overlap + " " + sentence + " "

if curr_chunk.strip():
    chunks.append(curr_chunk.strip())




# start=0
# while start < len(document):
#     end=start+chunk_size
#     chunk=document[start:end]
#     chunks.append(chunk)
#     start = end - chunk_overlap

print("Original document:")
print(document)

print("\nChunks:")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk}")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print("\nEmbedding information:")
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


import faiss
import numpy as np
#emb to float32
embeddings = np.asarray(embeddings, dtype="float32")
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("\nFAISS information:")
print("Vector dimension:", dimension)
print("Number of vectors:", index.ntotal)



query = "What is Python used for?"

#query to emb
query_embedding = model.encode([query])
query_embedding = np.asarray(query_embedding, dtype="float32")
faiss.normalize_L2(query_embedding)
k=2
scores, indices = index.search(query_embedding, k)

print("\nQuery:")
print(query)

print("\nTop results:")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    print(f"\nRank {rank}")
    print("Chunk:", chunks[idx])
    print("Score:", score)

retrieved_chunks=[]
for idx in indices[0]:
    retrieved_chunks.append(chunks[idx])

context="\n\n".join(retrieved_chunks)
print("\nContext for the query:")
print(context)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

print("\nGenerated Prompt:")
print(prompt)

#send prompt to llm
response=client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3-0324",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200
)
answer = response.choices[0].message.content

print("\nFinal Answer:")
print(answer)
# print("\nRaw response:")
# print(response)

# print("\nMessage:")
# print(response.choices[0].message)

# print("\nContent:")
# print(response.choices[0].message.content)