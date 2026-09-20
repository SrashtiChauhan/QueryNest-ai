import os
from huggingface_hub import InferenceClient
from src.config import LLM_MODEL


class LLM:
    def __init__(self):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError("HF_TOKEN not found in environment.")

        self.client = InferenceClient(token=token)
        self.model = LLM_MODEL

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200
        )

        return response.choices[0].message.content
