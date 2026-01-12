from pydantic import BaseModel
from typing import Optional, List, Literal


class Message(BaseModel):
    """Message in conversation (used in requests/responses)."""

    role: Literal["system", "user", "assistant"]
    content: str


class IntentInfo(BaseModel):
    intent: str
    concepts: List[str]


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "my-organization"


class CodeExtractionResult(BaseModel):
    complex_query: bool
    query_tokens: List[str]
    code_tokens: List[str]
    code_lines: List[str]
    query_lines: List[str]
    languages_detected: List[str]
    boost_query: Optional[str]
