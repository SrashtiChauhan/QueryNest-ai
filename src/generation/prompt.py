def build_prompt(context, question):
    """
    Build the grounded RAG prompt.
    """

    return f"""
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
