# Middleware for logging
import logging
from fastapi import Request

log = logging.getLogger(__name__)


async def log_requests(request: Request, call_next):
    log.debug(f"Request: {request.method} {request.url.path}")
    if request.method == "POST":
        log.debug(await request.json())

    response = await call_next(request)
    log.debug(f"Response: {response}")
    return response
