from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from models.domain import IntentInfo, Message, Model


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]


class ContextResponse(BaseModel):
    context: Message
    ctx_token_count: int
    raw_context: Optional[List[Dict]] = None
    system: Optional[Message] = None
    intent: Optional[IntentInfo] = None


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = 0
    message: Message
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None

    # Custom fields
    cached: bool = False
    retrieval_metadata: Optional[Dict[str, Any]] = None
