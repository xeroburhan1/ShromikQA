import os
import re
import json
import pickle
import logging
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def tokenize(text: str) -> list:
    """Simple tokenizer that extracts alphanumeric tokens (including Unicode for Bengali)."""
    return re.findall(r'\w+', text.lower())

def main():
    load_dotenv()
    
    base_dir = "d:/000 NSU 4th/cse 596/shromic QA/labour-law-assistant"
    chunks_path = os.path.join(base_dir, "data/processed/chunks.jsonl")
    bm25_path = os.path.join(base_dir, "data/processed/bm25_index.pkl")
    
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
        logger.error("No chunks found in chunks.jsonl.")
        return
        
    logger.info(f"Loaded {len(chunks)} chunks.")
    
    # Tokenize corpus
    logger.info("Tokenizing chunks for BM25...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    
    # Create BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Package BM25 with chunks metadata for retrieval
    index_data = {
        "bm25": bm25,
        "chunks": chunks
    }
    
    # Save index
    logger.info(f"Saving BM25 index to: {bm25_path}")
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, 'wb') as f:
        pickle.dump(index_data, f)
        
    logger.info("BM25 index build complete!")

if __name__ == "__main__":
    main()
