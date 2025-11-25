import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- Paths (using Path for Windows compatibility) ---
BASE_DIR = Path(__file__).parent.absolute()
DATA_PATH = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "college"

# --- Chunking function ---
def chunk_text(text, max_length=500):
    """Split text into smaller chunks for better retrieval"""
    # Split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Skip navigation-like text
        if len(para) < 50 or any(skip in para for skip in [
            'AboutHistory', 'Administration', 'Students', 'Facilities', 
            'Login', 'ERP', 'Downloads', 'CUET'
        ]):
            continue
            
        # If adding this paragraph exceeds max_length, save current chunk
        if len(current_chunk) + len(para) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# --- Ensure folders exist ---
DATA_PATH.mkdir(exist_ok=True)
CHROMA_PATH.mkdir(exist_ok=True)

# --- Load model ---
print("📥 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully.")

# --- Initialize Chroma client (Windows-compatible) ---
# Use string path for ChromaDB compatibility
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(
        anonymized_telemetry=False,  # Disable telemetry to avoid errors
        allow_reset=True
    )
)

# --- Delete and recreate collection ---
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"🗑️ Deleted old collection: '{COLLECTION_NAME}'")
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)
print(f"✨ Created new collection: '{COLLECTION_NAME}'")

# --- Load and chunk data ---
all_chunks = []
file_sources = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        file_path = DATA_PATH / file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                chunks = chunk_text(text, max_length=500)
                all_chunks.extend(chunks)
                file_sources.extend([file] * len(chunks))
                print(f"📄 {file}: {len(chunks)} chunks created")

if not all_chunks:
    print("❌ No valid chunks created. Add .txt files to the 'data' folder.")
    exit()

print(f"\n📊 Total chunks: {len(all_chunks)}")

# --- Add to Chroma ---
print("\n🔢 Creating embeddings and adding to ChromaDB...")
embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()
ids = [str(i) for i in range(len(all_chunks))]

# Add metadata with source file
metadatas = [{"source": source} for source in file_sources]

collection.add(
    documents=all_chunks, 
    embeddings=embeddings, 
    ids=ids,
    metadatas=metadatas
)

print(f"\n✅ Indexed {len(all_chunks)} chunks into ChromaDB at '{CHROMA_PATH}'")
print(f"📊 Total documents in collection: {collection.count()}")
print("\n🚀 Ready! Run: streamlit run app_improved.py")