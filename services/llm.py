import json
import logging

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict
import httpx

from models.domain import IntentInfo, Message
from models.request import ChatCompletionRequest
from models.response import ChatCompletionResponse
from config import settings

log = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.cheap_model = settings.CHEAP_MODEL
        self.top_model = settings.TOP_MODEL
        self.base_url = settings.OPENAI_API_BASE
        self.auth_token = f"{settings.LLM_AUTH_TYPE} {settings.OPENAI_API_KEY}"

        self.stats = {
            "intent_calls": 0,
            "generation_calls": 0,
            "total_tokens": 0,
            "errors": 0,
        }

    async def forward_request(self, chat_req: Dict):
        # Determine if streaming
        if chat_req["stream"]:
            return StreamingResponse(
                self._stream_openai_response(chat_req), media_type="text/event-stream"
            )
        else:
            return await self._forward_to_openai(chat_req)

    async def extract_intent(self, prompt: str) -> IntentInfo:
        intent_prompt = settings.INTENT_EXTRACTION_PROMPT + f"Request: {prompt}"
        try:
            completion_res = await self._forward_to_openai(
                ChatCompletionRequest(
                    model=settings.CHEAP_MODEL,
                    temperature=settings.INTENT_EXTRACTION_TEMP,
                    messages=[Message(role="user", content=intent_prompt)],
                ).dict(exclude_unset=True)
            )
            log.debug(f"Completion result:{completion_res}")
            return IntentInfo(**json.loads(completion_res.choices[0].message.content))
        except Exception as e:
            log.error(e)
            return IntentInfo(intent=prompt, concepts=[])

    async def _forward_to_openai(self, request_data: Dict) -> ChatCompletionResponse:
        """
        Forward non-streaming request to OpenAI
        """
        headers = {"Content-Type": "application/json", "Authorization": self.auth_token}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=request_data, headers=headers
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )

            return ChatCompletionResponse(**response.json())

    async def _stream_openai_response(self, request_data: Dict):
        """
        Stream response from OpenAI
        """
        headers = {"Content-Type": "application/json", "Authorization": self.auth_token}

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=request_data,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code, detail=error_text.decode()
                    )

                async for chunk in response.aiter_bytes():
                    yield chunk


# Singleton
llm_service = LLMService()
