# Middleware for logging
import logging
import json
from fastapi import Request

log = logging.getLogger(__name__)


async def log_requests(request: Request, call_next):
    log.debug(f"Request: {request.method} {request.url.path}")
    content_type = request.headers.get("Content-Type", "")
    if request.method == "POST" and "application/json" in content_type:
        try:
            body_bytes = await request.body()
            if body_bytes:
                body_str = body_bytes.decode("utf8")
                try:
                    parsed_req = json.loads(body_str)
                    log.debug(parsed_req)
                except json.JSONDecodeError:
                    log.debug(f"Body (Invalid JSON) {body_str}")
        except Exception as e:
            log.debug(f"Failed to log request {e}")

    response = await call_next(request)
    log.debug(f"Response: {response.status_code}")
    return response
