from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Create the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
    # max_new_tokens=100,
)

# Convert it into a ChatModel
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke("What are the strongest 5 countries in military power?")

# Print only the answer
print(result.content)