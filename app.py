import time

import streamlit as st

from src.document_loader import load_uploaded_pdf, split_documents
from src.embeddings import get_embeddings, check_embeddings_ready
from src.vector_store import create_vector_store, get_retriever
from src.rag_chain import ask_question, get_llm


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hybrid AI RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Hybrid AI RAG")
st.caption(
    "Local RAG performs document retrieval. "
    "Amazon Bedrock Agent with Amazon Nova generates the final answer."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Process documents
# ---------------------------------------------------------

def process_documents(files):
    """
    Local RAG pipeline:

    PDF
      ↓
    Load pages
      ↓
    Split into chunks
      ↓
    Local embeddings using Ollama
      ↓
    FAISS vector store
      ↓
    Retriever
    """

    pages = []

    for file in files:
        try:
            pages.extend(load_uploaded_pdf(file))
        except Exception as error:
            st.warning(
                f"Skipped {file.name}: {error}"
            )

    if not pages:
        raise ValueError(
            "No readable text was found in the uploaded PDFs."
        )

    # Split documents into chunks
    chunks = split_documents(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the uploaded PDFs."
        )

    # Local embeddings
    embeddings = get_embeddings()

    # Verify Ollama embedding model
    check_embeddings_ready(embeddings)

    # Create local FAISS vector store
    store = create_vector_store(
        chunks,
        embeddings,
    )

    # Create retriever
    retriever = get_retriever(
        store,
        k=3,
    )

    return retriever, len(pages), len(chunks)


# ---------------------------------------------------------
# Display sources
# ---------------------------------------------------------

def show_sources(sources):
    if not sources:
        return

    with st.expander(
        f"📚 Sources ({len(sources)})"
    ):
        for i, source in enumerate(
            sources,
            start=1,
        ):
            st.markdown(
                f"**[{i}] {source['file']} "
                f"- Page {source['page']}**"
            )

            st.caption(
                source["text"]
            )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
    ):

        if not uploaded:
            st.error(
                "Please upload at least one PDF first."
            )

        else:

            with st.spinner(
                "Processing documents locally..."
            ):

                try:

                    (
                        retriever,
                        n_pages,
                        n_chunks,
                    ) = process_documents(uploaded)

                    # Save retriever
                    st.session_state.retriever = retriever

                    # Reset chat for new documents
                    st.session_state.messages = []

                    # Reset LLM
                    st.session_state.llm = None

                    st.success(
                        f"Ready! {n_pages} pages "
                        f"and {n_chunks} chunks processed."
                    )

                except Exception as error:

                    st.session_state.retriever = None
                    st.session_state.llm = None

                    st.error(
                        f"Document processing failed: {error}"
                    )


# ---------------------------------------------------------
# Architecture information
# ---------------------------------------------------------

with st.sidebar:

    st.divider()

    st.subheader("⚙️ Architecture")

    st.markdown(
        """
        **Local System**

        📄 PDF  
        ↓  
        🔹 Chunking  
        ↓  
        🔹 Ollama Embeddings  
        ↓  
        🔹 FAISS  
        ↓  
        🔹 Retrieval  

        **AWS**

        Retrieved Context  
        ↓  
        ☁️ Amazon Bedrock Agent  
        ↓  
        🤖 Amazon Nova  
        ↓  
        💬 Final Answer
        """
    )

    st.divider()

    st.caption(
        "Ollama is used only for local embeddings. "
        "It does not generate the final answer."
    )


# ---------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if message.get("backend"):

            st.caption(
                f"Answered by: "
                f"{message['backend']}"
            )

        if message.get("model"):

            st.caption(
                f"Model: "
                f"{message['model']}"
            )

        if message.get("time") is not None:

            st.caption(
                f"Response time: "
                f"{message['time']:.2f} seconds"
            )

        if message.get("sources"):

            show_sources(
                message["sources"]
            )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # -----------------------------------------------------
    # Make sure documents are processed
    # -----------------------------------------------------

    if st.session_state.retriever is None:

        st.warning(
            "Please upload a PDF and click "
            "'Process Documents' first."
        )

    else:

        # -------------------------------------------------
        # Display user question
        # -------------------------------------------------

        with st.chat_message("user"):

            st.write(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Retrieving documents locally and "
                "generating answer with Amazon Bedrock..."
            ):

                try:

                    start_time = time.perf_counter()

                    # -------------------------------------
                    # Create Bedrock Agent LLM once
                    # -------------------------------------

                    if st.session_state.llm is None:

                        st.session_state.llm = get_llm()

                    # -------------------------------------
                    # Local retrieval + Bedrock generation
                    # -------------------------------------

                    result = ask_question(
                        question,
                        st.session_state.retriever,
                        st.session_state.llm,
                    )

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    # -------------------------------------
                    # Final answer
                    # -------------------------------------

                    st.write(
                        result["answer"]
                    )

                    # -------------------------------------
                    # Backend information
                    # -------------------------------------

                    st.caption(
                        "Answered by: "
                        "Amazon Bedrock Agent"
                    )

                    st.caption(
                        "Model: Amazon Nova"
                    )

                    st.caption(
                        f"Response time: "
                        f"{elapsed:.2f} seconds"
                    )

                    # -------------------------------------
                    # Sources
                    # -------------------------------------

                    if result.get("sources"):

                        show_sources(
                            result["sources"]
                        )

                    # -------------------------------------
                    # Save assistant response
                    # -------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get(
                                "sources",
                                [],
                            ),
                            "backend": (
                                "Amazon Bedrock Agent"
                            ),
                            "model": "Amazon Nova",
                            "time": elapsed,
                        }
                    )

                except ValueError as error:

                    st.error(
                        str(error)
                    )

                except Exception as error:

                    st.error(
                        "Amazon Bedrock failed. "
                        "No local LLM fallback was used."
                    )

                    st.exception(error)