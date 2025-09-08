from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:latest")
response = llm.invoke("Hello, what's up? tell me a joke.")
print(response)
