from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set Hugging Face cache location
os.environ["HF_HOME"] = "D:/huggingface_cache"

# Load a small Hugging Face model locally
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,
    },
)

# Convert to LangChain ChatModel
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke("Who is the father of Salman Khan?")

# Print response
print(result.content)