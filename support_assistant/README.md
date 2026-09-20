# Module 3: Zepto Grounded GenAI Support Assistant (`/support_assistant`)

## 1. Module Overview
This module implements a production-grade, grounded **GenAI Policy Support Assistant** for Zepto. The service allows customers to ask natural language questions regarding Zepto's delivery terms, cancellation windows, returns, membership tiers, and support hours.

The assistant is architected around:
- **Local Embedding & Vector Indexing**: SentenceTransformers (`all-MiniLM-L6-v2`) and persistent **ChromaDB** vector storage (zero API key, zero network dependency).
- **LangGraph StateGraph Workflow**: A 3-node graph with conditional intent routing (`classify_intent` $\rightarrow$ `retrieve_and_answer` or `direct_answer`).
- **Structured Pydantic Contract**: Enforcing strict JSON schema responses (`answer`, `sources`, `confidence`) with validation and retry-on-error logic.
- **FastAPI Service & Containerization**: High-performance `POST /ask` endpoint and a self-contained `Dockerfile`.
- **Deterministic Mock Baseline**: Gated via `MOCK_LLM` toggle (default `1`) ensuring 100% deterministic, offline grading without third-party API dependencies.

---

## 2. End-to-End RAG Architecture & Component Flow

```text
[ 1. Ingestion & Embedding ]
  8 Policy Docs (docs/doc_01.txt ... doc_08.txt)
             │
             ▼ (sentence-transformers / all-MiniLM-L6-v2)
  ChromaDB Vector Store (support_assistant/vector_store.py)
             │
═════════════╪═════════════════════════════════════════════════════════════════
[ 2. LangGraph Intent Router ] (support_assistant/graph.py)
             │
      User Query Ingestion (POST /ask)
             │
             ▼
      [Node 1: classify_intent]
             │
             ├─────────────────────────────────────────────────┐
             │ (policy_question)                               │ (general_question)
             ▼                                                 ▼
      [Node 2: retrieve_and_answer]                     [Node 3: direct_answer]
             │                                                 │
      • Real ChromaDB Cosine Top-3 Retrieval                   • Fixed canned response:
      • Generation:                                              "I can only answer questions
        - Mock: f"Based on the retrieved..."                      about Zepto policies right now."
        - Real LLM: Structured prompt + retry                  • sources = []
             │                                                 • confidence = 1.0
             └───────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                      [ Pydantic QueryResponse Schema ]
                      • answer: str
                      • sources: List[str]
                      • confidence: float (0.0 - 1.0)
```

### Detailed Stage Breakdown:
1. **Ingestion (`docs/`, `vector_store.py`)**:
   Loads the 8 Zepto policy files (`doc_01.txt` to `doc_08.txt`) covering delivery, returns, membership pass, tracking, order cancellations, damaged goods, gift cards, and support hours.
2. **Embedding (`vector_store.py`)**:
   Encodes each document chunk into 384-dimensional dense vectors using the local open-source `sentence-transformers/all-MiniLM-L6-v2` model and stores them in a persistent ChromaDB collection (`zepto_policies`) with cosine distance space.
3. **Intent Routing (`graph.py: classify_intent`)**:
   Evaluates incoming queries. In mock mode, a deterministic keyword heuristic matches domain terms (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`, `pass`, etc.) to route policy questions vs. general queries.
4. **Retrieval (`vector_store.py: retrieve`)**:
   Runs **for real in both mock and live modes**. Generates query embedding and performs cosine similarity top-3 nearest-neighbor search against ChromaDB.
5. **Generation & Schema Validation (`graph.py`, `schemas.py`)**:
   Produces the final grounded answer. Enforces Pydantic `QueryResponse` validation with retry-on-failure logic.

---

## 3. Mock Mode vs. Real LLM Mode (`MOCK_LLM` Toggle)

The pipeline is controlled by the `MOCK_LLM` environment variable:

| Pipeline Stage | Default Mock Mode (`MOCK_LLM=1` or unset) | Optional Real LLM Mode (`MOCK_LLM=0`) |
| :--- | :--- | :--- |
| **Intent Classification** | Keyword heuristic matching policy terms. No network call. | LLM zero-shot classification (`policy_question` vs `general_question`). |
| **Vector Retrieval** | **Real ChromaDB cosine search** (top-3 chunks). | **Real ChromaDB cosine search** (top-3 chunks). |
| **Policy Answer Generation** | Deterministic canned snippet: `f"Based on the retrieved context: {top_chunk[:200]}..."` with real source doc IDs. | Formats structured prompt and calls LLM (e.g. Groq LLaMA-3). Validates output with Pydantic; retries up to 2 times on validation error. |
| **General Query Response** | Fixed canned string: `"I can only answer questions about Zepto policies right now."` with `sources=[]`. | LLM generates polite redirect response without retrieval context. |

---

## 4. Structured Prompt Template (Role–Context–Task–Format–Length)

The prompt template (`support_assistant/prompts.py`) follows the complete prompt engineering skeleton with explicit negative constraints and few-shot grounding:

```text
### ROLE
You are Zepto's Customer Policy Support Assistant. You provide helpful, accurate, and customer-friendly policy assistance.

### CONTEXT
Below are the retrieved policy document chunks provided by the assignment:
---------------------
{context}
---------------------

### TASK
Answer the following customer question using ONLY the provided context above:
Customer Query: "{query}"

### EXPLICIT NEGATIVE CONSTRAINTS (MANDATORY)
1. Do NOT answer using any external information or assumptions not present in the provided context.
2. If the retrieved context does not contain sufficient facts to answer the question, state: "I cannot find this information in the provided Zepto policy documents."
3. Do NOT mention competitor services or fabricate policies, timeframes, or fee numbers.

### FORMAT
You must respond with a strictly valid JSON object matching the schema below:
{
  "answer": "<string: concise, direct answer>",
  "sources": ["<doc_id_1>", "<doc_id_2>"],
  "confidence": <float: 0.0 to 1.0>
}

### LENGTH
Keep your answer concise and direct (under 150 words).

### FEW-SHOT EXAMPLE
[Example Query]
"How much does priority delivery cost on Zepto?"

[Example Context]
[doc_01] Zepto delivers grocery and household essentials within 10 to 30 minutes. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15.

[Example Response]
{
  "answer": "Priority delivery is available at checkout for an additional fee of INR 15, which reserves the next available rider slot.",
  "sources": ["doc_01"],
  "confidence": 0.98
}
```

---

## 5. Recorded Example Calls & Raw JSON Transcripts (Mock Mode Baseline)

### Example 1: Policy Query (Triggering Real Retrieval & Grounding)
- **Endpoint**: `POST /ask`
- **Request Payload**:
```json
{
  "query": "Can I cancel my order once it is packed?"
}
```
- **Response JSON (HTTP 200 OK)**:
```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be...",
  "sources": [
    "doc_05",
    "doc_02",
    "doc_06"
  ],
  "confidence": 1.0
}
```
*Analysis*: Correctly routed to `retrieve_and_answer`. The vector search identified `doc_05` (Order Cancellation Policy) as the primary source with high cosine similarity.

---

### Example 2: General / Out-of-Domain Query (Triggering Direct Answer)
- **Endpoint**: `POST /ask`
- **Request Payload**:
```json
{
  "query": "What is the weather like in Mumbai today?"
}
```
- **Response JSON (HTTP 200 OK)**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
*Analysis*: Correctly classified as `general_question` and routed to `direct_answer` without triggering unnecessary vector lookups. `sources` is empty as expected.

---

## 6. Docker Containerization & Local Execution

### Build Docker Image:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
```

### Run Container Locally:
```bash
docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

### Test Serving Endpoint:
```bash
# Health check
curl http://localhost:7860/health

# Ask Policy Query
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee for orders under INR 149?"}'
```

---

## 7. Automated Test Suite Execution

Run the standalone verification suite:
```bash
python support_assistant/test_assistant.py
```
*(Verifies ChromaDB embedding generation, 3-node LangGraph routing, Pydantic JSON schema conformance, and FastAPI TestClient endpoints).*
