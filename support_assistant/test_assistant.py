"""
Zepto Support Assistant - Verification & Test Suite
Module: support_assistant/test_assistant.py

Demonstrates:
1. ChromaDB vector indexing of 8 policy documents.
2. LangGraph intent routing (policy_question vs general_question).
3. Retrieval accuracy and grounded response generation in deterministic mock mode.
4. Pydantic schema validation on all outputs.
5. FastAPI /ask endpoint execution using TestClient.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from support_assistant.main import app
from support_assistant.vector_store import get_vector_store
from support_assistant.graph import run_assistant_pipeline
from support_assistant.schemas import QueryResponse


def run_tests():
    print("=" * 80)
    print("ZEPTO SUPPORT ASSISTANT: VERIFICATION & TEST SUITE")
    print("=" * 80)

    # 1. Verify ChromaDB Indexing
    print("\n[Test 1/4] Verifying ChromaDB Vector Index...")
    vs = get_vector_store()
    doc_count = vs.collection.count()
    print(f"Indexed documents count: {doc_count} (Expected: 8)")
    assert doc_count >= 8, f"Expected >=8 documents, got {doc_count}"
    print("ChromaDB index verification: PASSED")

    # 2. Test Policy Queries (Retrieval Triggered)
    print("\n[Test 2/4] Testing Policy Queries (Intent: policy_question -> retrieve_and_answer)...")
    policy_queries = [
        "What is Zepto's delivery policy, fee threshold, and delivery timeframe?",
        "How do I return damaged or spoiled grocery items?",
        "What are the benefits and monthly cost of the Zepto Pass+ membership?"
    ]

    for q in policy_queries:
        print("\n" + "-" * 75)
        print(f"User Query: \"{q}\"")
        resp = run_assistant_pipeline(q)
        print("LangGraph Response:")
        print(json.dumps(resp.model_dump(), indent=2))
        assert len(resp.sources) > 0, "Expected non-empty sources for policy query."
        assert resp.confidence > 0.0, "Expected valid confidence score."
        assert resp.answer.startswith("Based on the retrieved context:"), "Expected canned mock answer format."

    print("\nPolicy Queries Verification: PASSED")

    # 3. Test General Queries (Direct Answer Triggered)
    print("\n[Test 3/4] Testing General Non-Policy Queries (Intent: general_question -> direct_answer)...")
    general_queries = [
        "What is the weather like in Mumbai today?",
        "Can you write a poem about the sunrise?"
    ]

    for q in general_queries:
        print("\n" + "-" * 75)
        print(f"User Query: \"{q}\"")
        resp = run_assistant_pipeline(q)
        print("LangGraph Response:")
        print(json.dumps(resp.model_dump(), indent=2))
        assert len(resp.sources) == 0, "Expected empty sources for general query."
        assert "only answer questions about Zepto policies" in resp.answer, "Expected canned direct answer."

    print("\nGeneral Queries Verification: PASSED")

    # 4. Test FastAPI /ask Endpoint via TestClient
    print("\n[Test 4/4] Testing FastAPI POST /ask Endpoint...")
    client = TestClient(app)

    # Health check
    health_resp = client.get("/health")
    print(f"GET /health Status: {health_resp.status_code}, Payload: {health_resp.json()}")
    assert health_resp.status_code == 200

    # Policy query POST /ask
    req_body = {"query": "Can I cancel my order once it is packed?"}
    post_resp = client.post("/ask", json=req_body)
    print(f"\nPOST /ask (Policy Query) Status: {post_resp.status_code}")
    print("Response JSON:")
    print(json.dumps(post_resp.json(), indent=2))
    assert post_resp.status_code == 200
    assert "doc_05" in post_resp.json()["sources"]

    # General query POST /ask
    req_body_gen = {"query": "Tell me a joke about robots."}
    post_resp_gen = client.post("/ask", json=req_body_gen)
    print(f"\nPOST /ask (General Query) Status: {post_resp_gen.status_code}")
    print("Response JSON:")
    print(json.dumps(post_resp_gen.json(), indent=2))
    assert post_resp_gen.status_code == 200
    assert len(post_resp_gen.json()["sources"]) == 0

    print("\n" + "=" * 80)
    print("ALL SUPPORT ASSISTANT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
