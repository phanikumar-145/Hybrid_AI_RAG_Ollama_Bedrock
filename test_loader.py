from src.document_loader import load_pdf

pages = load_pdf("data/sample.pdf")

print("Total pages loaded:", len(pages))
print("\n--- Metadata of page 1 ---")
print(pages[0].metadata)
print("\n--- First 300 characters of page 1 ---")
print(pages[0].page_content[:300])