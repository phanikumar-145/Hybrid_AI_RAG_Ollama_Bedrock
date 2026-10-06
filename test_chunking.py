from src.document_loader import load_pdf, split_documents

pages = load_pdf("data/sample.pdf")
chunks = split_documents(pages)

print("Pages loaded:", len(pages))
print("Chunks created:", len(chunks))

print("\n--- Chunk 1 ---")
print(chunks[0].page_content)
print("Metadata:", chunks[0].metadata["source"], "| page index:", chunks[0].metadata["page"])
print("Length:", len(chunks[0].page_content), "characters")

print("\n--- Chunk 2 ---")
print(chunks[1].page_content)