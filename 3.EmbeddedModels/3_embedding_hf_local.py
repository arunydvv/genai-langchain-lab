from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load embedding model locally
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert text into embeddings
text = "I love eating pizza."

vector = embedding_model.embed_query(text)

print(vector)
print("Dimensions:", len(vector))