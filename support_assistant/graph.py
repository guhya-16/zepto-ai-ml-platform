"""
Zepto Support Assistant - LangGraph Orchestration Graph
Module: support_assistant/graph.py

Implements a LangGraph StateGraph with 3 nodes:
- classify_intent: Keyword heuristic / LLM classifier
- retrieve_and_answer: Real ChromaDB vector retrieval + grounded response generation
- direct_answer: Non-policy direct response

Gated by MOCK_LLM environment variable (default: 1 / unset -> fully deterministic mock mode).
Includes retry-on-failure logic for real LLM mode.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import ValidationError

from langgraph.graph import StateGraph, END
from support_assistant.schemas import QueryResponse
from support_assistant.vector_store import get_vector_store
from support_assistant.prompts import POLICY_RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Keywords for deterministic mock intent classification
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
    "pin code", "pass", "damaged", "missing", "wallet"
]


class AssistantState(TypedDict):
    """LangGraph TypedDict State Definition."""
    query: str
    intent: Optional[str]  # "policy_question" or "general_question"
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float
    retry_count: int
    raw_llm_output: Optional[str]


def is_mock_mode() -> bool:
    """Check MOCK_LLM environment variable (default: True)."""
    val = os.getenv("MOCK_LLM", "1").strip().lower()
    return val not in ("0", "false", "no")


# ============================================================================
# LangGraph Node 1: classify_intent
# ============================================================================
def classify_intent_node(state: AssistantState) -> Dict[str, Any]:
    """
    Classifies incoming query as 'policy_question' or 'general_question'.
    - Mock Mode (default): Uses deterministic keyword matching heuristic.
    - Real Mode: Invokes LLM intent classifier.
    """
    query = state["query"].lower().strip()
    mock_active = is_mock_mode()

    if mock_active:
        # Keyword-based deterministic classification
        is_policy = any(kw in query for kw in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
        logger.info(f"[Node: classify_intent] (Mock Mode) Query: '{state['query']}' -> Intent: {intent}")
        return {"intent": intent}
    else:
        # Optional Real LLM Mode classification
        try:
            import openai  # or httpx to groq
            client = openai.OpenAI(
                base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", "mock-key"))
            )
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "llama3-8b-8192"),
                messages=[
                    {"role": "system", "content": "Classify the user query into either 'policy_question' (questions about delivery, returns, refunds, pass, cancellations, damaged items, tracking, gift cards, support hours) or 'general_question'. Output only the intent label."},
                    {"role": "user", "content": state["query"]}
                ],
                temperature=0.0
            )
            raw_intent = resp.choices[0].message.content.strip().lower()
            intent = "policy_question" if "policy" in raw_intent else "general_question"
            return {"intent": intent}
        except Exception as e:
            logger.warning(f"Real LLM classification failed ({e}). Falling back to keyword heuristic.")
            is_policy = any(kw in query for kw in POLICY_KEYWORDS)
            return {"intent": "policy_question" if is_policy else "general_question"}


# ============================================================================
# LangGraph Node 2: retrieve_and_answer (For policy_question)
# ============================================================================
def retrieve_and_answer_node(state: AssistantState) -> Dict[str, Any]:
    """
    Executes real vector retrieval from ChromaDB and generates grounded response.
    - Retrieval: Always runs for real via cosine similarity top-3.
    - Generation:
        - Mock Mode (default): Returns formatted snippet template from top-1 chunk.
        - Real Mode: Prompts LLM with structured Role-Context-Task template and validates with Pydantic.
    """
    query = state["query"]
    vector_store = get_vector_store()

    # Step 1: Real vector retrieval from ChromaDB
    retrieved_chunks = vector_store.retrieve(query=query, top_k=3)
    source_ids = [chunk["id"] for chunk in retrieved_chunks]

    mock_active = is_mock_mode()

    if mock_active:
        # Deterministic Mock Generation
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]["content"]
            # Extract first ~200 characters cleanly up to a sentence boundary
            snippet = top_chunk[:200].strip()
            if len(top_chunk) > 200:
                snippet += "..."
            answer_text = f"Based on the retrieved context: {snippet}"
        else:
            answer_text = "Based on the retrieved context: No relevant Zepto policy documents were found."

        logger.info(f"[Node: retrieve_and_answer] (Mock Mode) Sources: {source_ids}")
        return {
            "retrieved_chunks": retrieved_chunks,
            "answer": answer_text,
            "sources": source_ids,
            "confidence": 1.0
        }
    else:
        # Optional Real LLM Generation with Pydantic Validation & Retry
        context_str = "\n\n".join([f"[{c['id']}] {c['content']}" for c in retrieved_chunks])
        prompt = POLICY_RAG_PROMPT_TEMPLATE.format(context=context_str, query=query)

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                import openai
                client = openai.OpenAI(
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    api_key=os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", "mock-key"))
                )
                resp = client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "llama3-8b-8192"),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt if attempt == 0 else f"{prompt}\n\nCORRECTION: Your previous output failed JSON validation. Ensure valid JSON matching {{\"answer\": str, \"sources\": list[str], \"confidence\": float}}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                raw_json = json.loads(resp.choices[0].message.content)
                validated = QueryResponse(**raw_json)
                return {
                    "retrieved_chunks": retrieved_chunks,
                    "answer": validated.answer,
                    "sources": validated.sources,
                    "confidence": validated.confidence,
                    "retry_count": attempt
                }
            except (ValidationError, Exception) as err:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {err}")
                if attempt == max_retries:
                    # Graceful fallback on retry exhaustion
                    top_chunk = retrieved_chunks[0]["content"][:200] if retrieved_chunks else "Zepto policy documents"
                    return {
                        "retrieved_chunks": retrieved_chunks,
                        "answer": f"Based on the retrieved context: {top_chunk}",
                        "sources": source_ids,
                        "confidence": 0.80,
                        "retry_count": attempt
                    }


# ============================================================================
# LangGraph Node 3: direct_answer (For general_question)
# ============================================================================
def direct_answer_node(state: AssistantState) -> Dict[str, Any]:
    """
    Handles general non-policy queries.
    - Mock Mode: Returns fixed canned response.
    - Real Mode: Direct LLM generation without retrieval.
    """
    mock_active = is_mock_mode()

    if mock_active:
        canned_msg = "I can only answer questions about Zepto policies right now."
        logger.info(f"[Node: direct_answer] (Mock Mode) Canned output returned.")
        return {
            "retrieved_chunks": [],
            "answer": canned_msg,
            "sources": [],
            "confidence": 1.0
        }
    else:
        try:
            import openai
            client = openai.OpenAI(
                base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", "mock-key"))
            )
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "llama3-8b-8192"),
                messages=[
                    {"role": "system", "content": "You are Zepto's customer assistant. Politely inform the user you specialize in Zepto delivery, return, and membership policies."},
                    {"role": "user", "content": state["query"]}
                ],
                temperature=0.3
            )
            return {
                "retrieved_chunks": [],
                "answer": resp.choices[0].message.content.strip(),
                "sources": [],
                "confidence": 0.90
            }
        except Exception:
            return {
                "retrieved_chunks": [],
                "answer": "I can only answer questions about Zepto policies right now.",
                "sources": [],
                "confidence": 1.0
            }


# ============================================================================
# Conditional Edge Router
# ============================================================================
def route_intent(state: AssistantState) -> str:
    """Routes state based on classified intent."""
    intent = state.get("intent", "general_question")
    if intent == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"


# ============================================================================
# Build LangGraph StateGraph
# ============================================================================
def build_assistant_graph():
    """Constructs and compiles the 3-node LangGraph workflow."""
    workflow = StateGraph(AssistantState)

    # Add Nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Conditional Routing Edge
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    # Node completions -> END
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


# Compiled singleton graph
assistant_graph = build_assistant_graph()


def run_assistant_pipeline(query: str) -> QueryResponse:
    """Executes the complete compiled LangGraph workflow for an input query."""
    initial_state: AssistantState = {
        "query": query,
        "intent": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "retry_count": 0,
        "raw_llm_output": None
    }

    final_state = assistant_graph.invoke(initial_state)

    # Validate output against Pydantic schema
    response = QueryResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )
    return response
