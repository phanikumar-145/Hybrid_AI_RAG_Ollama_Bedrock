from src.prompts import RAG_PROMPT, NOT_FOUND_MESSAGE
from src.bedrock_llm import (
    invoke_bedrock_agent,
    get_bedrock_llm,
    check_bedrock_llm_ready,
)


def format_context(docs):
    """Join retrieved document chunks into one context block."""

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_sources(docs):
    """Create source information for the UI or terminal."""

    sources = []

    for doc in docs:

        page = doc.metadata.get("page", 0) + 1

        sources.append(
            {
                "file": doc.metadata.get(
                    "source",
                    "Unknown file",
                ),
                "page": page,
                "text": doc.page_content,
            }
        )

    return sources


def build_agent_prompt(question, context):
    """
    Create the prompt sent to the Bedrock Agent.

    The Agent should answer only from the retrieved
    document context.
    """

    prompt = RAG_PROMPT.format(
        context=context,
        question=question,
    )

    return prompt


def ask_question(question, retriever, agent):
    """
    Complete hybrid RAG pipeline.

    Local:
        Question
          ↓
        FAISS retrieval
          ↓
        Retrieved context

    AWS:
        Context + Question
          ↓
        Bedrock Agent
          ↓
        Amazon Nova
          ↓
        Final answer
    """

    # --------------------------------------------------
    # 1. Validate question
    # --------------------------------------------------

    if not question or len(question.strip()) < 3:
        raise ValueError(
            "Please type a valid question "
            "(at least 3 characters)."
        )

    # --------------------------------------------------
    # 2. Local retrieval
    # --------------------------------------------------

    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # --------------------------------------------------
    # 3. Prepare context
    # --------------------------------------------------

    context = format_context(docs)

    # --------------------------------------------------
    # 4. Build prompt
    # --------------------------------------------------

    prompt = build_agent_prompt(
        question,
        context,
    )

    # --------------------------------------------------
    # 5. Send to Bedrock Agent
    # --------------------------------------------------

    answer = invoke_bedrock_agent(
        agent,
        prompt,
    )

    answer = answer.strip()

    # --------------------------------------------------
    # 6. Handle not-found answer
    # --------------------------------------------------

    if NOT_FOUND_MESSAGE.lower() in answer.lower():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # --------------------------------------------------
    # 7. Return answer + sources
    # --------------------------------------------------

    return {
        "answer": answer,
        "sources": build_sources(docs),
    }


def get_llm():
    """
    Return the Bedrock Agent configuration.

    Kept as get_llm() so the existing app/terminal
    code can continue using the same function name.
    """

    agent = get_bedrock_llm()

    check_bedrock_llm_ready(agent)

    return agent