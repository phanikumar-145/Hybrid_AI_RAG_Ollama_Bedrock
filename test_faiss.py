from src.document_loader import load_pdf, split_documents
from src.embeddings import get_embeddings, check_embeddings_ready
from src.vector_store import create_vector_store

# Load and chunk the PDF
pages = load_pdf("data/sample.pdf")
chunks = split_documents(pages)

# Embed and store in FAISS
embeddings = get_embeddings()
check_embeddings_ready(embeddings)
vector_store = create_vector_store(chunks, embeddings)
print("Vector store created with", vector_store.index.ntotal, "vectors\n")

# Search by meaning
question = "Which cloud platform was used for deployment?"
results = vector_store.similarity_search_with_score(question, k=2)

for doc, score in results:
    print(f"Distance: {score:.3f} | {doc.metadata['source']} | page index {doc.metadata['page']}")
    print(doc.page_content[:200])
    print("-" * 40)