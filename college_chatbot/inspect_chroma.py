import chromadb

client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_collection("college")

print("Collections:", client.list_collections())
print("Documents:", collection.get(ids=None))