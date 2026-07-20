import os
import json
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    load_dotenv()
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    base_dir = "d:/000 NSU 4th/cse 596/shromic QA/labour-law-assistant"
    chunks_path = os.path.join(base_dir, "data/processed/chunks.jsonl")
    chroma_db_dir = os.path.join(base_dir, "data/processed/chroma_db")
    
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found: {chunks_path}. Please run chunk_sections.py first.")
        return
        
    logger.info(f"Loading chunks from: {chunks_path}")
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    if not chunks:
        logger.error("No chunks found in the chunks.jsonl file.")
        return
        
    logger.info(f"Loaded {len(chunks)} chunks.")
    
    # Initialize embedding model
    logger.info(f"Loading embedding model: {embedding_model_name}")
    try:
        model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model {embedding_model_name}: {e}")
        return
        
    # Extract text, ids, and metadata
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        # ChromaDB metadata can only contain string, int, float, or bool
        meta = {
            "chapter": chunk.get("chapter", "Unknown"),
            "section_number": str(chunk.get("section_number", "")),
            "section_title": chunk.get("section_title", ""),
            "source_doc": chunk.get("source_doc", ""),
            "page_number": int(chunk.get("page_number", 0)),
            "law_version_date": chunk.get("law_version_date", "")
        }
        metadatas.append(meta)
        
    # Generate embeddings
    logger.info("Generating embeddings for chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}.")
    
    # Initialize ChromaDB persistent client
    logger.info(f"Initializing ChromaDB in directory: {chroma_db_dir}")
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
    
    # Create or get collection
    collection_name = "labour_law_collection"
    # Delete collection if it exists to build fresh
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing ChromaDB collection: {collection_name}")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )
    
    # Add to collection in batches of 100 to prevent issues
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        logger.info(f"Adding batch {i//batch_size + 1} ({i} to {min(i+batch_size, len(chunks))})...")
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        
    logger.info("ChromaDB vector store build complete!")

if __name__ == "__main__":
    main()
