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