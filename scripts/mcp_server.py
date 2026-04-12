#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP


SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "tolkative-rag-mcp")
CONTEXT_URL = os.environ.get("CTX_URL", "http://localhost:8000/context")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("CTX_TIMEOUT", "30"))

mcp = FastMCP(SERVER_NAME)


@mcp.tool(
    name="get_ton_context",
    description="This tool allows to query TON documentation for additional info",
)
async def get_context(query: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": query}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(CONTEXT_URL, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        return {
            "ok": False,
            "error": "context_endpoint_error",
            "status_code": exc.response.status_code,
            "detail": detail,
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "context_request_failed",
            "detail": str(exc),
        }

    body = response.json()
    return {
        "ok": True,
        "context": body.get("context", ""),
        "ctx_token_count": body.get("ctx_token_count", 0),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
