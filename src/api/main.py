import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generate_answer import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(override=True)

app = FastAPI(
    title="Bangladesh Labour Law LLM Assistant API",
    description="Backend API for retrieving and generating answers grounded in Bangladesh labour law.",
    version="1.0.0"
)

# Initialize retrieval components
base_dir = "d:/000 NSU 4th/cse 596/shromic QA/labour-law-assistant"
retriever = HybridRetriever(base_dir=base_dir)
reranker = Reranker()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    retrieved_chunks: list

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    logger.info(f"Received question: '{question}'")
    
    try:
        # 1. Retrieve candidates from ChromaDB and BM25
        raw_candidates = retriever.retrieve(question, top_k=5)
        
        # 2. Re-rank retrieved candidates using Cross-Encoder
        retrieved_chunks = reranker.rerank(question, raw_candidates, top_k=3)
        
        # 3. Generate structured answer grounding it on retrieved chunks
        answer = generate_answer(retrieved_chunks, question)
        
        return AnswerResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
