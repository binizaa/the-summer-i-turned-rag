import chromadb
import uuid

# Create client and collection
##client = chromadb.HttpClient(host="localhost", port=8000)


client = chromadb.CloudClient(
  api_key='ck-5uLGRzkv6nQsjudAACG69mtjVR3upykfuS1w74E9SXae',
  tenant='e9f6326d-d9a1-4bf4-a0fd-3ff0813408ce',
  database='roborregos'
)

collection = client.get_or_create_collection(name="chisme_corporativo")

with open("chisme_corporativo.txt", "r", encoding="utf-8") as f:
    chisme_corporativo = f.read().splitlines()

documents = [line for line in chisme_corporativo if line.strip()]

ids = [str(uuid.uuid4()) for _ in range(len(documents))]

metadatas = [{"line": i} for i in range(len(documents))]

# Add documents to collection
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

results = collection.query(
    query_texts=[
        "Qué es chisme corporativo?"
    ],
    n_results=5
)

for i, query_results in enumerate(results["documents"]):
    print(f"\nQuery {i}")
    print("\n".join(query_results))