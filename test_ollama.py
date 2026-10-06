from src.local_llm import get_local_llm, check_local_llm_ready

llm = get_local_llm()
check_local_llm_ready(llm)
print("Ollama is ready.\n")

reply = llm.invoke("In one short sentence, what is Amazon Bedrock?")
print("Reply from local model:")
print(reply.content)