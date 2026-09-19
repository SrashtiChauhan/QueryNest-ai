import os
from huggingface_hub import InferenceClient


class LLM:
    def __init__(self):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError("HF_TOKEN not found in environment.")

        self.client = InferenceClient(token=token)
        self.model = "deepseek-ai/DeepSeek-V3-0324"

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
