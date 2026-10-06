from src.document_loader import load_pdf, split_documents
from src.embeddings import get_embeddings, check_embeddings_ready
from src.vector_store import create_vector_store, get_retriever

# Build the vector store (Steps 5 to 7)
pages = load_pdf("data/sample.pdf")
chunks = split_documents(pages)
embeddings = get_embeddings()
check_embeddings_ready(embeddings)
vector_store = create_vector_store(chunks, embeddings)

# Step 8: create the retriever
retriever = get_retriever(vector_store, k=3)

questions = [
    "Which cloud platform was used for deployment?",
    "What is the candidate's education?",
]

for question in questions:
    print("QUESTION:", question)
    docs = retriever.invoke(question)
    print("Chunks retrieved:", len(docs))
    for i, doc in enumerate(docs, start=1):
        print(f"  [{i}] {doc.metadata['source']} | page index {doc.metadata['page']}")
        print("      ", doc.page_content[:120].replace("\n", " "))
    print()