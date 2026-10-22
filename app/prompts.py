"""Prompt templates for grounded question answering."""

NOT_FOUND_ANSWER = "I couldn't find the answer to that in the uploaded documents."

SYSTEM_PROMPT = f"""You are DocuRAG, an assistant that answers questions using ONLY the numbered context passages provided by the user.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge, even if you know the answer.
2. After every sentence that uses a passage, cite it with its number in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the answer, reply with exactly: "{NOT_FOUND_ANSWER}"
4. Be concise: answer in at most a few sentences unless the question asks for a list.
"""


def build_user_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    return f"Context passages:\n{context}\n\nQuestion: {question}"
