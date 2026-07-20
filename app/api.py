import os
import sys
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from rag.orchestrator import RAGOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Shromik QA - Bangladesh Labour Law RAG API",
    version="2.0.0"
)


# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
def startup_event():
    global orchestrator
    index_path = os.path.join(base_dir, "data/processed/faiss_index.bin")
    meta_path = os.path.join(base_dir, "data/processed/faiss_metadata.json")
    orchestrator = RAGOrchestrator(index_path=index_path, meta_path=meta_path)
    logger.info("RAG Orchestrator initialized for FastAPI server.")

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    start_time = time.time()
    res = orchestrator.answer_question(req.query.strip(), top_k=req.top_k)
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return {
        "query": res["query"],
        "answer": res["answer"],
        "retrieved_chunks": res["retrieved_chunks"],
        "num_chunks": len(res["retrieved_chunks"]),
        "execution_time_ms": elapsed_ms
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Shromic AI Legal RAG", "faiss_vectors": 374}

# Serve Static Web Frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return "<h1>Shromic AI Legal Server Active</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
