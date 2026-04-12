from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
from models.domain import Message


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None

    # max_tokens is for GPT-4o/Turbo, max_completion_tokens is for o1/o3 models
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    stream: Optional[bool] = False
    # Required if you want token usage stats while streaming
    stream_options: Optional[Dict[str, Any]] = None

    # --- Modern Tool Use (Replaces Functions) ---
    tools: Optional[List[Dict[str, Any]]] = None
    # tool_choice can be 'auto', 'required', 'none' or a specific tool dict
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True

    # --- Advanced Features ---
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    user: Optional[str] = None

    # Ensures deterministic usage (reproducibility)
    seed: Optional[int] = None

    # Usage: {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}
    response_format: Optional[Dict[str, Any]] = None

    # Enterprise/Organization features
    service_tier: Optional[str] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None


class ContextRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = None
