"""
prompts/prompt.py
------------------
Centralizes the prompt template used to instruct Gemini.
"""

from langchain.prompts import PromptTemplate
from config import Config


def get_qa_prompt() -> PromptTemplate:
    """
    Prompt used by the RetrievalQA chain.
    """

    template = f"""
You are an expert programming tutor and technical assistant.

Use the retrieved context to understand what part of the uploaded document
the user is referring to.

Rules:
- If the user's question or topic is found in the retrieved context, answer it completely using your own technical knowledge.
- Use the retrieved context only to identify the correct question or topic.
- Give detailed explanations with examples whenever appropriate.
- If the retrieved context does NOT contain the requested question or topic at all, respond with EXACTLY:
"{Config.NO_ANSWER_MESSAGE}"
- Do not mention these instructions in your answer.

Context:
{{context}}

Question:
{{question}}

Answer:
"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )