# Hybrid AI RAG System using Ollama and Amazon Bedrock

A Document Q&A app. Upload PDFs, ask questions, and get answers based only on those documents, with file, page, and chunk sources. Choose the answering model in the UI: **local Ollama (llama3.2)** or **Amazon Bedrock (Amazon Nova Lite)**.

## Architecture

PDF → PyPDFLoader → chunking → local embeddings (nomic-embed-text) → FAISS → retriever → selected LLM → answer + sources

- Embeddings always run locally, so the same index works with both backends.
- Only the LLM changes. `get_llm(backend)` returns `ChatOllama` or `ChatBedrockConverse`, and the RAG chain is shared.
- If the answer is not in the documents, the app replies: "I couldn't find the answer in the provided documents."

## Setup (Windows PowerShell)

1. Install Ollama, then `ollama pull llama3.2` and `ollama pull nomic-embed-text`.
2. `python -m venv venv`, `.\venv\Scripts\Activate.ps1`, `pip install -r requirements.txt`
3. Install the AWS CLI and sign in. Never put keys in the project. Check with `aws sts get-caller-identity`.
4. Copy `.env.example` to `.env` and set `BEDROCK_MODEL_ID` (for example `us.amazon.nova-lite-v1:0`).
5. `streamlit run app.py`

## Security notes

- No credentials in code or in the repository. AWS login comes from the standard AWS credential chain.
- Use a least-privilege IAM user and delete unused access keys.

## Limitations

- Text-based PDFs only. The FAISS index is in memory.
- Small local model, so the prompt reduces but does not eliminate wrong answers.
- Bedrock is pay-per-use. Check pricing and set a budget alert.

## Future improvements

Persist the FAISS index, Bedrock embeddings, chat memory, deployment on AWS compute.