import logging
from fastapi import APIRouter, HTTPException
from models.domain import Message
from models.request import ChatCompletionRequest, ContextRequest
from models.response import ContextResponse, MessageContextResponse
from core.retrieval import InputError, pull_context
import json

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/chat_context", response_model=MessageContextResponse)
async def rag_completion_context(request: ChatCompletionRequest):
    """
    Context retrieval endpoint
    """
    try:
        log.debug(
            f"Context requested: {json.dumps(request.model_dump(exclude_unset=True))}"
        )
        res_ctx = await pull_context(
            request.messages, request.max_completion_tokens or request.max_tokens
        )
        return res_ctx
    except InputError as e:
        raise HTTPException(
            status_code=400, detail={"error": {"code": "invalid_input", "msg": str(e)}}
        )
    except Exception as e:
        log.error(str(e))
        raise HTTPException(status_code=500, detail="Failed to pull context")


@router.post("/context", response_model=ContextResponse)
async def raw_context(request: ContextRequest):
    try:
        res_ctx = await pull_context(
            [Message(role="user", content=request.query)], request.max_tokens
        )
        return ContextResponse(
            context=res_ctx.context, ctx_token_count=res_ctx.ctx_token_count
        )
    except InputError as e:
        raise HTTPException(
            status_code=400, detail={"error": {"code": "invalid_input", "msg": str(e)}}
        )
    except Exception as e:
        log.error(str(e))
        raise HTTPException(status_code=500, detail="Failed to pull context")
