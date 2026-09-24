"""Request id and access log for every HTTP call."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
  """Attach X-Request-ID and write one access line per response."""

  async def dispatch(self, request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s %s %.1fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


def request_id_from(request: Request) -> str | None:
  return getattr(request.state, "request_id", None)
