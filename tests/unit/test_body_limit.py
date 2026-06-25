from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from reference_eta.serving.body_limit import RequestBodyLimitMiddleware


def _scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/test",
        "raw_path": b"/test",
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def _receive_from(chunks: list[bytes]) -> Callable[[], Awaitable[dict[str, object]]]:
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive() -> dict[str, object]:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    return receive


async def _run(
    middleware: RequestBodyLimitMiddleware,
    *,
    scope: dict[str, object],
    chunks: list[bytes],
) -> tuple[list[dict[str, object]], bool]:
    sent: list[dict[str, object]] = []
    downstream_called = False

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    original = middleware.app

    async def wrapped_app(scope_value, receive, send_value):  # noqa: ANN001
        nonlocal downstream_called
        downstream_called = True
        await original(scope_value, receive, send_value)

    middleware.app = wrapped_app
    await middleware(scope, _receive_from(chunks), send)
    return sent, downstream_called


def _reader_app() -> Callable[..., Awaitable[None]]:
    async def app(scope, receive, send):  # noqa: ANN001
        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return app


def _status(messages: list[dict[str, object]]) -> int:
    return int(
        next(message["status"] for message in messages if message["type"] == "http.response.start")
    )


def test_chunked_body_without_content_length_is_enforced() -> None:
    middleware = RequestBodyLimitMiddleware(_reader_app(), limit_provider=lambda: 5)
    messages, called = asyncio.run(_run(middleware, scope=_scope(), chunks=[b"123", b"456"]))
    assert not called
    assert _status(messages) == 413


def test_declared_oversized_body_is_rejected_before_downstream() -> None:
    middleware = RequestBodyLimitMiddleware(_reader_app(), limit_provider=lambda: 5)
    messages, called = asyncio.run(
        _run(
            middleware,
            scope=_scope([(b"content-length", b"6")]),
            chunks=[b"123456"],
        )
    )
    assert not called
    assert _status(messages) == 413


def test_invalid_size_configuration_is_server_error() -> None:
    def invalid_limit() -> int:
        raise ValueError("bad config")

    middleware = RequestBodyLimitMiddleware(_reader_app(), limit_provider=invalid_limit)
    messages, called = asyncio.run(_run(middleware, scope=_scope(), chunks=[b"{}"]))
    assert not called
    assert _status(messages) == 503


def test_invalid_or_negative_content_length_is_client_error() -> None:
    for value in (b"not-a-number", b"-1"):
        middleware = RequestBodyLimitMiddleware(_reader_app(), limit_provider=lambda: 100)
        messages, called = asyncio.run(
            _run(middleware, scope=_scope([(b"content-length", value)]), chunks=[b"{}"])
        )
        assert not called
        assert _status(messages) == 400


def test_streamed_body_at_limit_reaches_downstream() -> None:
    middleware = RequestBodyLimitMiddleware(_reader_app(), limit_provider=lambda: 6)
    messages, called = asyncio.run(_run(middleware, scope=_scope(), chunks=[b"123", b"456"]))
    assert called
    assert _status(messages) == 204


def test_liveness_style_get_bypasses_invalid_body_limit_configuration() -> None:
    async def live_app(scope, receive, send):  # noqa: ANN001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    def invalid_limit() -> int:
        raise ValueError("bad config")

    middleware = RequestBodyLimitMiddleware(live_app, limit_provider=invalid_limit)
    scope = _scope()
    scope["method"] = "GET"
    messages, called = asyncio.run(_run(middleware, scope=scope, chunks=[]))
    assert called
    assert _status(messages) == 200
