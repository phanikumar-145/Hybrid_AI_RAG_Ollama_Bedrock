from langchain_core.prompts import ChatPromptTemplate

# The exact sentence the app shows when the answer is not in the documents
NOT_FOUND_MESSAGE = "I couldn't find the answer in the provided documents."

# {context}, {question} and {not_found} are filled in later
_template = """You are a helpful assistant that answers questions using ONLY the context below.

Rules:
1. Use only the information in the context. Do not use outside knowledge.
2. If the answer is not clearly in the context, reply with exactly: {not_found}
3. Keep the answer clear and short.

Context:
{context}

Question: {question}

Answer:"""

# partial() fills {not_found} now, so later we only pass context and question
RAG_PROMPT = ChatPromptTemplate.from_template(_template).partial(
    not_found=NOT_FOUND_MESSAGE
)