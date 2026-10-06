from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    """Embed all chunks and store them in a FAISS index."""

    if not chunks:
        raise ValueError("There are no chunks to store.")

    try:
        # Embeds every chunk and builds the FAISS index in one step.
        # The chunk text and metadata (source, page) are stored too.
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as error:
        raise RuntimeError(f"Could not create the vector store: {error}")

    return vector_store

def get_retriever(vector_store, k=3):
    """Turn the vector store into a retriever that returns the top-k chunks."""

    return vector_store.as_retriever(
        search_type="similarity",   # plain similarity search
        search_kwargs={"k": k},     # how many chunks to return
    )