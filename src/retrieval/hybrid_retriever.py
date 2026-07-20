import os
import re
import pickle
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())

class HybridRetriever:
    def __init__(self, base_dir: str = "d:/000 NSU 4th/cse 596/shromic QA/labour-law-assistant"):
        load_dotenv()
        self.base_dir = base_dir
        self.chroma_db_dir = os.path.join(self.base_dir, "data/processed/chroma_db")
        self.bm25_path = os.path.join(self.base_dir, "data/processed/bm25_index.pkl")
        
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        # Lazy load indexes to speed up import times
        self.model = None
        self.chroma_collection = None
        self.bm25 = None
        self.chunks_dict = {}
        
    def _initialize(self):
        if self.model is not None:
            return
            
        logger.info("Initializing Hybrid Retriever indexes...")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        
        # Load ChromaDB
        logger.info(f"Connecting to ChromaDB at: {self.chroma_db_dir}")
        chroma_client = chromadb.PersistentClient(path=self.chroma_db_dir)
        self.chroma_collection = chroma_client.get_collection("labour_law_collection")
        
        # Load BM25 index
        logger.info(f"Loading BM25 index from: {self.bm25_path}")
        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError(f"BM25 index file not found at {self.bm25_path}. Please run build_bm25_index.py first.")
            
        with open(self.bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.chunks = bm25_data["chunks"]
            # Map chunk_id to chunks for O(1) lookup
            self.chunks_dict = {chunk["chunk_id"]: chunk for chunk in self.chunks}
            
    def vector_search(self, query: str, top_k: int = 10) -> list:
        self._initialize()
        query_embedding = self.model.encode(query).tolist()
        
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        vector_results = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                score = results["distances"][0][idx] if results["distances"] else 0.0
                # ChromaDB distance score (typically L2 or cosine distance)
                vector_results.append({
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx]
                })
        return vector_results

    def bm25_search(self, query: str, top_k: int = 10) -> list:
        self._initialize()
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort and get top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        bm25_results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            # Skip chunks with zero score if query tokens have no match
            if scores[idx] <= 0:
                continue
            bm25_results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[idx]),
                "text": chunk["text"],
                "metadata": {
                    "chapter": chunk.get("chapter", "Unknown"),
                    "section_number": str(chunk.get("section_number", "")),
                    "section_title": chunk.get("section_title", ""),
                    "source_doc": chunk.get("source_doc", ""),
                    "page_number": int(chunk.get("page_number", 0)),
                    "law_version_date": chunk.get("law_version_date", "")
                }
            })
        return bm25_results

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list:
        """
        Retrieves top_k documents using Reciprocal Rank Fusion (RRF) 
        combining vector search and BM25 search.
        """
        self._initialize()
        
        vector_results = self.vector_search(query, top_k=10)
        bm25_results = self.bm25_search(query, top_k=10)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Apply vector ranks
        for rank, item in enumerate(vector_results):
            chunk_id = item["chunk_id"]
            # rank is 0-indexed, so add 1 for 1-based rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Apply BM25 ranks
        for rank, item in enumerate(bm25_results):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Sort by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        
        merged_results = []
        for chunk_id in sorted_chunk_ids:
            chunk = self.chunks_dict[chunk_id]
            merged_results.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "rrf_score": rrf_scores[chunk_id],
                "metadata": {
                    "chapter": chunk.get("chapter", "Unknown"),
                    "section_number": str(chunk.get("section_number", "")),
                    "section_title": chunk.get("section_title", ""),
                    "source_doc": chunk.get("source_doc", ""),
                    "page_number": int(chunk.get("page_number", 0)),
                    "law_version_date": chunk.get("law_version_date", "")
                }
            })
            
        return merged_results

if __name__ == "__main__":
    retriever = HybridRetriever()
    print("Hybrid retriever structure initialized.")
