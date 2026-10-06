from src.document_loader import load_pdf, split_documents
from src.embeddings import get_embeddings, check_embeddings_ready
from src.vector_store import create_vector_store, get_retriever
from src.local_llm import get_local_llm, check_local_llm_ready
from src.rag_chain import ask_question

# Build the pipeline
pages = load_pdf("data/sample.pdf")
chunks = split_documents(pages)
embeddings = get_embeddings()
check_embeddings_ready(embeddings)
vector_store = create_vector_store(chunks, embeddings)
retriever = get_retriever(vector_store, k=3)

llm = get_local_llm()
check_local_llm_ready(llm)

questions = [
    "Which cloud platform was used for deployment?",   # answer is in the PDF
    "Who won the 2018 FIFA World Cup?",                # NOT in the PDF
    "",                                                # invalid question
]

for question in questions:
    print("QUESTION:", repr(question))
    try:
        result = ask_question(question, retriever, llm)
        print("ANSWER:", result["answer"])
        if result["sources"]:
            for i, source in enumerate(result["sources"], start=1):
                print(f"  [{i}] {source['file']} | page {source['page']}")
        else:
            print("SOURCES: none")
    except ValueError as error:
        print("ERROR:", error)
    print()