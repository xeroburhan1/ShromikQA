import os
import sys
import json
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def build_faiss_index(chunks_path: str, index_output_path: str, meta_output_path: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads section chunks, generates vector embeddings using SentenceTransformer, 
    normalizes embeddings, and builds a FAISS IndexFlatIP (Cosine Similarity) index.
    Saves index binary and JSON metadata mapping.
    """
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file missing: {chunks_path}")
        sys.exit(1)
        
    logger.info(f"Loading chunks from: {chunks_path}")
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    if not chunks:
        logger.error("No chunks found in JSONL file.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(chunks)} chunks.")
    
    logger.info(f"Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [f"Section {c.get('section_number', '')}. {c.get('section_title', '')}\n{c.get('text', '')}" for c in chunks]
    
    logger.info("Generating embeddings for all section chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize vectors to L2 norm=1 for exact Cosine Similarity using Inner Product (IP) index
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    
    logger.info(f"Building FAISS IndexFlatIP (dimension {dimension})...")
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    
    logger.info(f"FAISS index contains {faiss_index.ntotal} vectors.")
    
    # Save FAISS index binary
    clean_index_path = os.path.normpath(index_output_path)
    clean_meta_path = os.path.normpath(meta_output_path)
    
    os.makedirs(os.path.dirname(clean_index_path), exist_ok=True)
    faiss.write_index(faiss_index, clean_index_path)
    logger.info(f"Saved FAISS index to: {clean_index_path}")

    
    # Save metadata JSON file
    metadata_list = []
    for idx, c in enumerate(chunks):
        metadata_list.append({
            "faiss_id": idx,
            "chunk_id": c.get("chunk_id", ""),
            "chapter": c.get("chapter", "Unknown"),
            "section_number": str(c.get("section_number", "")),
            "section_title": c.get("section_title", ""),
            "text": c.get("text", ""),
            "source_doc": c.get("source_doc", ""),
            "page_number": c.get("page_number", 1),
            "law_version_date": c.get("law_version_date", "")
        })
        
    os.makedirs(os.path.dirname(meta_output_path), exist_ok=True)
    with open(clean_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved FAISS metadata store to: {clean_meta_path}")


def main():
    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    chunks_path = "data/processed/chunks.jsonl"
    index_path = "data/processed/faiss_index.bin"
    meta_path = "data/processed/faiss_metadata.json"
    
    build_faiss_index(chunks_path, index_path, meta_path, model_name=model_name)


if __name__ == "__main__":
    main()
