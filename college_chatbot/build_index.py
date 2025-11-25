import os
from sentence_transformers import SentenceTransformer
import chromadb

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "college"

# --- Ensure folders exist ---
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- Load model ---
print("🔄 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully.")

# --- Initialize Chroma client ---
client = chromadb.PersistentClient(path=CHROMA_PATH)

# --- Create or get existing collection ---
existing_collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"📚 Using existing collection: '{COLLECTION_NAME}'")
else:
    collection = client.create_collection(COLLECTION_NAME)
    print(f"✨ Created new collection: '{COLLECTION_NAME}'")

# --- Load data ---
docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        file_path = os.path.join(DATA_PATH, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append(text)
                print(f"📄 Loaded: {file} ({len(text)} characters)")
            else:
                print(f"⚠️ Skipped empty file: {file}")

if not docs:
    print("❌ No .txt files found in data/. Add files and try again.")
    exit()

# --- Add to Chroma ---
print("\n🔢 Creating embeddings and adding to ChromaDB...")
embeddings = model.encode(docs, show_progress_bar=True).tolist()
ids = [str(i) for i in range(len(docs))]

collection.add(documents=docs, embeddings=embeddings, ids=ids)

print(f"\n✅ Indexed {len(docs)} documents into ChromaDB at '{CHROMA_PATH}'", flush=True)
print(f"📊 Total documents now in collection: {collection.count()}", flush=True)