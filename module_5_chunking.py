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
chunk_overlap=20
chunks=[]
start=0
while start < len(document):
    end=start+chunk_size
    chunk=document[start:end]
    chunks.append(chunk)
    start = end - chunk_overlap

print("Original document:")
print(document)

print("\nChunks:")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk}")