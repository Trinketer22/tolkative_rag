import logging
import time
from uuid import uuid4
import json
from fastapi import APIRouter, HTTPException, Header
from models.domain import Message
from models.request import ChatCompletionRequest
from models.response import ChatCompletionResponse, ChatCompletionChoice
from config import settings
from typing import Optional
from core.retrieval import InputError, pull_context
from services.llm import llm_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions")
async def proxy_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Proxy endpoint that adds context before forwarding to OpenAI
    """
    try:
        # Inject context into messages
        log.debug(
            f"Completion requested: {json.dumps(request.model_dump(exclude_unset=True))}"
        )
        request_ctx = await pull_context(request.messages)
        if request_ctx.ctx_token_count == 0:
            return ChatCompletionResponse(
                id=str(uuid4()),
                created=int(time.time()),
                model=settings.PROXY_MODEL_NAME,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=settings.PROXY_MODEL_CONTEXT_NOT_FOUND_MESSAGE,
                        ),
                    )
                ],
            )
        # Prepare the request for OpenAI
        openai_request = request.dict(exclude_unset=True)
        openai_request["messages"][-1] = request_ctx.context.model_dump(
            exclude_unset=True
        )
        openai_request["model"] = settings.TOP_MODEL
        if request_ctx.system is not None:
            openai_request["messages"].insert(
                0, request_ctx.system.model_dump(exclude_unset=True)
            )

        return await llm_service.forward_request(openai_request)

    except InputError as e:
        raise HTTPException(
            status_code=400, detail={"error": {"code": "invalid_input", "msg": str(e)}}
        )
    except Exception as e:
        log.error(str(e))
        raise HTTPException(status_code=500, detail="Failed to proxy request")
