import os
import sys
import re
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

# Regex to match explicit section number mentions, e.g. "Section 27", "sec 100", "section 33"
SECTION_REF_PATTERN = re.compile(r'(?:section|sec\.?|धारा)\s*#?\s*(\d{1,3}[a-z]?)', re.IGNORECASE)

class FAISSRetriever:
    def __init__(self, index_path: str = "data/processed/faiss_index.bin", meta_path: str = "data/processed/faiss_metadata.json"):
        load_dotenv()
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        self.model = None
        self.faiss_index = None
        self.metadata = []
        self.section_map = {}
        
    def _lazy_load(self):
        if self.faiss_index is None:
            logger.info(f"Loading FAISS index from {self.index_path}...")
            if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
                raise FileNotFoundError("FAISS index or metadata missing. Run python index/build_faiss_index.py first.")
                
            self.faiss_index = faiss.read_index(self.index_path)
            
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            self.section_map = {}
            for item in self.metadata:
                sec_num = str(item.get("section_number", "")).strip()
                if sec_num:
                    if sec_num not in self.section_map:
                        self.section_map[sec_num] = []
                    self.section_map[sec_num].append(item)
                    
            logger.info(f"Loaded FAISS index ({self.faiss_index.ntotal} vectors) and {len(self.metadata)} metadata records.")

        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)


    def extract_section_references(self, query: str) -> list:
        """Extract explicit section numbers from raw query string."""
        matches = SECTION_REF_PATTERN.findall(query)
        return list(dict.fromkeys(matches))

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Retrieves top-k relevant statutory section chunks.
        Performs explicit section-number regex override to prioritize exact citations,
        followed by dense FAISS vector search.
        """
        self._lazy_load()
        
        override_chunks = []
        explicit_secs = self.extract_section_references(query)
        
        if explicit_secs:
            logger.info(f"Found explicit section reference(s) in query: {explicit_secs}")
            for sec in explicit_secs:
                if sec in self.section_map:
                    for chunk in self.section_map[sec]:
                        override_chunk = dict(chunk)
                        override_chunk["retrieval_source"] = "section_number_override"
                        override_chunk["score"] = 1.0
                        override_chunks.append(override_chunk)
                        
        # Dense FAISS vector search
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.faiss_index.search(query_vec, top_k * 2)
        
        vector_chunks = []
        if len(indices) > 0:
            for rank, faiss_id in enumerate(indices[0]):
                if faiss_id < 0 or faiss_id >= len(self.metadata):
                    continue
                chunk = dict(self.metadata[faiss_id])
                chunk["retrieval_source"] = "faiss_dense"
                chunk["score"] = float(distances[0][rank])
                vector_chunks.append(chunk)
                
        # Merge override + dense chunks, preserving order and removing duplicates
        combined = []
        seen_ids = set()
        
        for c in override_chunks + vector_chunks:
            cid = c.get("chunk_id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                combined.append(c)
                if len(combined) >= top_k:
                    break
                    
        return combined

if __name__ == "__main__":
    retriever = FAISSRetriever()
    res = retriever.retrieve("What is the notice period under Section 27?", top_k=3)
    print(f"Retrieved {len(res)} chunks:")
    for r in res:
        print(f" - [{r['retrieval_source']}] Sec {r['section_number']}: {r['section_title']} (score: {r['score']:.4f})")
