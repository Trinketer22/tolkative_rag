import logging
from fastapi import APIRouter, HTTPException
from models.response import ContextResponse, MessageContextResponse
from core.retrieval import InputError, pull_context
import json

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/context", response_model=ContextResponse)
async def rag_chat(request: ChatCompletionRequest):
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
    except InputError as e:
        raise HTTPException(
            status_code=400, detail={"error": {"code": "invalid_input", "msg": str(e)}}
        )
    except Exception as e:
        log.error(str(e))
        raise HTTPException(status_code=500, detail="Failed to pull context")
