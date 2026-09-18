"""
Zepto Support Assistant - FastAPI Application Service
Module: support_assistant/main.py

Exposes:
- POST /ask : Receives QueryRequest and returns Pydantic-validated QueryResponse.
- GET /health : Service status and ChromaDB vector count.
"""

import os
import sys
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support_assistant.schemas import QueryRequest, QueryResponse
from support_assistant.graph import run_assistant_pipeline, is_mock_mode
from support_assistant.vector_store import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Zepto AI Policy Support Assistant",
    description="Grounded GenAI Policy Support Assistant powered by LangGraph, ChromaDB, and FastAPI.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Warm up ChromaDB vector store and index documents on startup."""
    logger.info("Initializing Zepto Policy Assistant Vector Store...")
    vs = get_vector_store()
    mode = "MOCK_LLM (Deterministic)" if is_mock_mode() else "REAL_LLM (Live API)"
    logger.info(f"Assistant Service Started in [{mode}] mode. Indexed {vs.collection.count()} policy chunks.")


@app.get("/")
def root():
    """Root metadata endpoint."""
    return {
        "service": "Zepto AI Policy Support Assistant",
        "status": "online",
        "mock_mode": is_mock_mode(),
        "docs_endpoint": "/docs",
        "ask_endpoint": "/ask"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    vs = get_vector_store()
    return {
        "status": "healthy",
        "mock_llm": is_mock_mode(),
        "indexed_documents": vs.collection.count()
    }


@app.post("/ask", response_model=QueryResponse)
def ask_policy_assistant(request: QueryRequest) -> QueryResponse:
    """
    POST /ask:
    Accepts user question, routes intent via LangGraph, performs vector retrieval from ChromaDB,
    and returns grounded response conforming to Pydantic QueryResponse schema.
    """
    try:
        response = run_assistant_pipeline(request.query)
        return response
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal assistant processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("support_assistant.main:app", host="0.0.0.0", port=port, reload=False)
