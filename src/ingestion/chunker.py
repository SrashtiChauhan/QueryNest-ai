from src.config import CHUNK_SIZE

def chunk_document(document, chunk_size=CHUNK_SIZE):
    """
    Split a document into sentence-aware chunks
    with sentence-level overlap.
    """

    # Split document into sentences
    sentences = [
        sentence.strip()
        for sentence in document.split(".")
        if sentence.strip()
    ]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence + "."

        # Add sentence if it fits
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "

        else:
            # Save current chunk
            chunks.append(current_chunk.strip())

            # Keep the last complete sentence as overlap
            previous_sentences = current_chunk.strip().split(".")
            previous_sentences = [
                s.strip()
                for s in previous_sentences
                if s.strip()
            ]

            if previous_sentences:
                overlap = previous_sentences[-1] + "."
            else:
                overlap = ""

            current_chunk = overlap + " " + sentence + " "

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

