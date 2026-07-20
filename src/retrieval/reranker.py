import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        load_dotenv()
        self.model_name = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = None
        
    def _initialize(self):
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}. Fallback to native scoring will be used.")
            self.model = False
            
    def rerank(self, query: str, candidates: list, top_k: int = 3) -> list:
        """
        Reranks a list of retrieved candidate chunks for a query using the Cross-Encoder.
        Returns top_k chunks sorted by relevance.
        """
        self._initialize()
        
        if not candidates:
            return []
            
        # If model loading failed, return original candidates up to top_k
        if self.model is False or self.model is None:
            logger.warning("Reranker not initialized. Returning raw candidates.")
            return candidates[:top_k]
            
        # Pair query with each candidate's text
        pairs = [[query, item["text"]] for item in candidates]
        
        try:
            scores = self.model.predict(pairs)
            
            # Attach scores to items
            for idx, score in enumerate(scores):
                candidates[idx]["rerank_score"] = float(score)
                
            # Sort descending by rerank score
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Returning raw candidates.")
            return candidates[:top_k]

if __name__ == "__main__":
    reranker = Reranker()
    print("Reranker structure initialized.")
