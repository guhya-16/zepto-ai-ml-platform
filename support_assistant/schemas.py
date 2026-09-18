"""
Zepto Support Assistant - Request/Response Schemas
Module: support_assistant/schemas.py
"""

from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request schema."""
    query: str = Field(..., description="The user question or inquiry about Zepto policies.", min_length=1)


class QueryResponse(BaseModel):
    """Structured response schema for Zepto policy assistant."""
    answer: str = Field(..., description="The grounded response answering the query.")
    sources: List[str] = Field(default_factory=list, description="List of source document or chunk IDs used.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
