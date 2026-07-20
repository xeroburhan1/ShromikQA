import os
import sys
import json
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from rag.retriever import FAISSRetriever
from rag.generator import generate_answer

class RAGOrchestrator:
    def __init__(self, index_path: str = "data/processed/faiss_index.bin", meta_path: str = "data/processed/faiss_metadata.json"):
        self.retriever = FAISSRetriever(index_path=index_path, meta_path=meta_path)
        
    def answer_question(self, query: str, top_k: int = 5) -> dict:
        """
        Processes a user question end-to-end:
        1. Dense retrieval + Section-Number Override via FAISS
        2. Context formatting
        3. LLM generation with section grounding
        4. Returns answer string + list of cited statutory section chunks
        """
        logger.info(f"Orchestrating response for query: '{query}'")
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        answer = generate_answer(retrieved_chunks, query)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks_retrieved": len(retrieved_chunks)
        }

def main():
    load_dotenv()
    if len(sys.argv) < 2:
        query = "What is the notice period for terminating a permanent worker under Section 27?"
    else:
        query = " ".join(sys.argv[1:])
        
    orchestrator = RAGOrchestrator()
    result = orchestrator.answer_question(query)
    
    print("\n" + "="*60)
    print(f"QUERY: {result['query']}")
    print("="*60)
    print(result['answer'])
    print("\n" + "-"*60)
    print("CITED STATUTORY CONTEXT CHUNKS:")
    for idx, chunk in enumerate(result['retrieved_chunks']):
        print(f"[{idx+1}] Section {chunk['section_number']}: {chunk['section_title']} (Source: {chunk['retrieval_source']})")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
