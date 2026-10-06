import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path, original_name=None):
    """Read a PDF from disk and return a list of Documents (one per page)."""

    # Check the file exists before trying to read it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # PyPDFLoader reads the PDF and creates one Document per page
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Remove pages that have no text (blank or scanned-image pages)
    pages = [page for page in pages if page.page_content.strip()]

    # If nothing is left, the PDF is empty or has no readable text
    if not pages:
        raise ValueError(
            "This PDF has no readable text. It may be empty or a scanned image."
        )

    # Replace the file path in metadata with the real file name
    name = original_name or os.path.basename(file_path)
    for page in pages:
        page.metadata["source"] = name

    return pages


def load_uploaded_pdf(uploaded_file):
    """Load a PDF uploaded through Streamlit (used later in the UI step)."""

    # PyPDFLoader needs a file path, but Streamlit gives us a file in memory.
    # So we save it to a temporary file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        return load_pdf(tmp_path, original_name=uploaded_file.name)
    finally:
        # Always delete the temporary file, even if loading fails
        os.remove(tmp_path)

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split page Documents into smaller chunk Documents."""

    # Create the splitter with our size settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split every page into chunks.
    # Each chunk keeps the metadata (source, page) of the page it came from.
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the documents.")

    return chunks