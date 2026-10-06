from langchain_ollama import OllamaEmbeddings

from src import config


def get_embeddings():
    """Create the embedding model that talks to the local Ollama server."""
    return OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL)


def check_embeddings_ready(embeddings):
    """Do a tiny test so we can show a clear error if Ollama has a problem."""
    try:
        embeddings.embed_query("test")
    except Exception as error:
        message = str(error).lower()

        # The model has not been downloaded yet
        if "not found" in message:
            raise RuntimeError(
                f"Model '{config.OLLAMA_EMBED_MODEL}' is not installed. "
                f"Run: ollama pull {config.OLLAMA_EMBED_MODEL}"
            )

        # Ollama is not running
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Open the Ollama app and check http://localhost:11434"
        )