import chromadb

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "college"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

print("✅ Connected to ChromaDB")
print("📦 Collection name:", collection.name)
print("📄 Total documents:", collection.count())