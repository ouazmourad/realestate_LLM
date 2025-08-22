import os
from typing import Any, Optional


class LLM:
    """LLM wrapper that uses a dummy model unless a transformers model is specified."""

    def __init__(self) -> None:
        self.model_name = os.getenv("RAG_MODEL_NAME")
        self.pipeline: Optional[Any] = None
        if self.model_name:
            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:  # pragma: no cover - import error path
                raise RuntimeError(
                    "transformers package is required when RAG_MODEL_NAME is set"
                ) from exc
            self.pipeline = pipeline("text2text-generation", model=self.model_name)

    def generate(self, question: str, context: str) -> str:
        if self.pipeline:
            prompt = (
                "Answer the question using only the provided context.\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            result = self.pipeline(prompt, max_new_tokens=128)[0]["generated_text"]
            return result.strip()
        # Deterministic dummy response for tests/dev
        return f"Answer: {context.strip()}"
