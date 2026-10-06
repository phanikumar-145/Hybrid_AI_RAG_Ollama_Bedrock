from langchain_ollama import ChatOllama

from src import config


def get_local_llm():
    """Create the local chat model (Ollama)."""
    return ChatOllama(
        model=config.OLLAMA_LLM_MODEL,   # model name comes from .env / config.py
        temperature=0,                   # least random, best for factual answers
    )


def check_local_llm_ready(llm):
    """Send a tiny message and show a clear error if Ollama has a problem."""
    try:
        llm.invoke("Hi")
    except Exception as error:
        message = str(error).lower()

        # The model has not been downloaded yet
        if "not found" in message:
            raise RuntimeError(
                f"Model '{config.OLLAMA_LLM_MODEL}' is not installed. "
                f"Run: ollama pull {config.OLLAMA_LLM_MODEL}"
            )

        # Ollama is not running
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Open the Ollama app and check http://localhost:11434"
        )