"""
Zepto Support Assistant - Vector Store & Embedding Engine
Module: support_assistant/vector_store.py

Ingests 8 Zepto policy documents, chunks and embeds them locally using
all-MiniLM-L6-v2 via sentence-transformers, and indexes them in ChromaDB.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parent / "docs"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


class LocalVectorStore:
    """Manages local document embeddings and ChromaDB vector retrieval."""

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading local SentenceTransformer model: {MODEL_NAME}...")
        self.embedder = SentenceTransformer(MODEL_NAME)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Ensure documents are indexed
        self._index_documents_if_needed()

    def _index_documents_if_needed(self):
        """Loads and indexes all 8 policy documents if collection is empty."""
        current_count = self.collection.count()
        if current_count >= 8:
            logger.info(f"ChromaDB collection already contains {current_count} documents. Reusing index.")
            return

        logger.info(f"Indexing policy documents from {DOCS_DIR} into ChromaDB...")
        doc_files = sorted(list(DOCS_DIR.glob("doc_*.txt")))

        if not doc_files:
            raise FileNotFoundError(f"No document files found in {DOCS_DIR}")

        documents = []
        ids = []
        metadatas = []

        for doc_path in doc_files:
            doc_id = doc_path.stem  # e.g. 'doc_01'
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            documents.append(content)
            ids.append(doc_id)
            metadatas.append({"source": doc_id, "file_name": doc_path.name})

        # Generate embeddings
        embeddings = self.embedder.encode(documents, normalize_embeddings=True).tolist()

        # Add to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Successfully indexed {len(ids)} documents into ChromaDB collection '{COLLECTION_NAME}'.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar document chunks for a query using cosine similarity.
        """
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i in range(len(docs)):
                # Chroma cosine distance = 1 - cosine_similarity
                similarity = 1.0 - distances[i]
                retrieved.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i],
                    "similarity": round(similarity, 4)
                })

        logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:40]}...' (Top match: {retrieved[0]['id'] if retrieved else 'None'})")
        return retrieved


# Global vector store instance
_vector_store_instance = None


def get_vector_store() -> LocalVectorStore:
    """Singleton provider for vector store."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = LocalVectorStore()
    return _vector_store_instance
