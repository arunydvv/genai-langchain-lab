from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Initialize embedding model
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=300
)

# Documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# User query
query = "Tell me about Virat Kohli"

# Generate embeddings for documents
doc_embeddings = embedding.embed_documents(documents)

# Generate embedding for query
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Find document with highest similarity
index, score = max(
    enumerate(scores),
    key=lambda x: x[1]
)

# Display result
print("Query:", query)
print("\nMost Similar Document:")
print(documents[index])
print("\nSimilarity Score:", score)