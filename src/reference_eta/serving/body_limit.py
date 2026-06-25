from __future__ import annotations

from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Buffer bounded JSON request bodies so chunked uploads cannot bypass the byte cap."""

    _BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, app: ASGIApp, *, limit_provider: Callable[[], int]) -> None:
        self.app = app
        self.limit_provider = limit_provider

    @staticmethod
    def _headers(scope: Scope) -> dict[bytes, list[bytes]]:
        result: dict[bytes, list[bytes]] = {}
        for name, value in scope.get("headers", []):
            result.setdefault(name.lower(), []).append(value)
        return result

    @staticmethod
    def _content_length(headers: dict[bytes, list[bytes]]) -> int | None:
        values = headers.get(b"content-length", [])
        if not values:
            return None
        if len(values) != 1:
            raise ValueError("Request must contain at most one Content-Length header")
        try:
            value = int(values[0].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Content-Length must be a nonnegative integer") from error
        if value < 0:
            raise ValueError("Content-Length must be a nonnegative integer")
        return value

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = self._headers(scope)
        method = str(scope.get("method", "GET")).upper()
        has_declared_body = b"content-length" in headers or b"transfer-encoding" in headers
        if method not in self._BODY_METHODS and not has_declared_body:
            await self.app(scope, receive, send)
            return

        try:
            maximum_bytes = int(self.limit_provider())
        except (TypeError, ValueError):
            response = JSONResponse(
                status_code=503,
                content={"detail": "Serving request-size configuration is invalid"},
            )
            await response(scope, receive, send)
            return

        try:
            content_length = self._content_length(headers)
        except ValueError as error:
            response = JSONResponse(status_code=400, content={"detail": str(error)})
            await response(scope, receive, send)
            return
        if content_length is not None and content_length > maximum_bytes:
            response = JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body is {content_length} bytes; "
                        f"configured maximum is {maximum_bytes}"
                    )
                },
            )
            await response(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > maximum_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body exceeded the configured maximum of {maximum_bytes} bytes"
                        )
                    },
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)
