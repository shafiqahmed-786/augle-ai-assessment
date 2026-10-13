"""Local mock LLM HTTP servers speaking the Anthropic Messages and OpenAI chat-completions protocols.

These are TEST DOUBLES: a tiny scripted state machine (call a tool first, then answer using the
tool result it was sent). They exist to verify our provider code, request/response schemas and the
agent loop over real HTTP. They are not language models and prove nothing about model behaviour.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockState:
    """Mutable knobs and a log of what the server received."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.problems: list[str] = []
        self.always_tool = False  # never answer unless tool use is disabled
        self.bad_arguments = False  # emit invalid tool arguments / unknown tool
        self.fail_statuses: list[int] = []  # pop one per request before behaving normally
        self.tool_name = "analyze_bbox_areas"
        self.tool_args: dict[str, Any] = {"max_examples": 3}


def _anthropic_reply(state: MockState, body: dict[str, Any]) -> dict[str, Any]:
    msgs = body["messages"]
    ids_used: set[str] = set()
    for m in msgs:
        if m["role"] not in ("user", "assistant"):
            state.problems.append(f"bad role {m['role']}")
        if isinstance(m["content"], list):
            for b in m["content"]:
                if b["type"] == "tool_use":
                    ids_used.add(b["id"])
                if b["type"] == "tool_result" and b["tool_use_id"] not in ids_used:
                    state.problems.append("tool_result without matching tool_use")
    for t in body.get("tools", []):
        if set(t) != {"name", "description", "input_schema"}:
            state.problems.append(f"bad tool shape {sorted(t)}")
    tools_disabled = body.get("tool_choice") == {"type": "none"}
    last = msgs[-1]
    saw_result = (
        isinstance(last["content"], list) and last["content"] and last["content"][0].get("type") == "tool_result"
    )
    if tools_disabled:
        return {
            "content": [{"type": "text", "text": "MOCK-LLM forced final answer (tool_choice none)"}],
            "stop_reason": "end_turn",
        }
    if saw_result and not state.always_tool:
        payload = json.loads(last["content"][0]["content"])
        facts = (
            f"ok={payload['ok']} flagged={payload['result']['suspiciously_small_count']}"
            if payload["ok"]
            else f"tool error: {payload['error']}"
        )
        return {
            "content": [{"type": "text", "text": f"MOCK-LLM final answer; tool said {facts}"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    name, args = ("does_not_exist", {}) if state.bad_arguments else (state.tool_name, state.tool_args)
    if state.bad_arguments:
        args = {"abs_area_threshold_px": "huge"}
        name = "analyze_bbox_areas"
    return {
        "content": [
            {"type": "text", "text": "calling tool"},
            {"type": "tool_use", "id": f"toolu_{len(state.requests)}", "name": name, "input": args},
        ],
        "stop_reason": "tool_use",
    }


def _openai_reply(state: MockState, body: dict[str, Any]) -> dict[str, Any]:
    msgs = body["messages"]
    if msgs[0]["role"] != "system":
        state.problems.append("first message must be system")
    pending: set[str] = set()
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                pending.add(tc["id"])
                if not isinstance(tc["function"]["arguments"], str):
                    state.problems.append("function.arguments must be a JSON string")
        if m["role"] == "tool" and m["tool_call_id"] not in pending:
            state.problems.append("tool message without matching tool_call")
    for t in body.get("tools", []):
        if t.get("type") != "function" or set(t["function"]) != {"name", "description", "parameters"}:
            state.problems.append("bad tool shape")
    if body.get("tool_choice") == "none":
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "MOCK-LLM forced final answer (tool_choice none)"},
                    "finish_reason": "stop",
                }
            ]
        }
    last = msgs[-1]
    if last["role"] == "tool" and not state.always_tool:
        payload = json.loads(last["content"])
        facts = (
            f"ok={payload['ok']} flagged={payload['result']['suspiciously_small_count']}"
            if payload["ok"]
            else f"tool error: {payload['error']}"
        )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": f"MOCK-LLM final answer; tool said {facts}"},
                    "finish_reason": "stop",
                }
            ]
        }
    args = json.dumps({"abs_area_threshold_px": "huge"} if state.bad_arguments else state.tool_args)
    if state.bad_arguments and len(state.requests) % 2 == 0:
        args = "{not json"
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{len(state.requests)}",
                            "type": "function",
                            "function": {"name": state.tool_name, "arguments": args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def start_mock_llm(kind: str) -> tuple[str, MockState, Any]:
    """Start a server on a free port; returns ``(base_url, state, stop_callable)``."""
    state = MockState(kind)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            state.headers.append({k.lower(): v for k, v in self.headers.items()})
            state.requests.append(body)
            if state.fail_statuses:
                code = state.fail_statuses.pop(0)
                payload = json.dumps({"error": {"message": "mock failure"}}).encode()
                self.send_response(code)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if kind == "anthropic":
                ok = (
                    self.path == "/v1/messages"
                    and self.headers.get("x-api-key") == "test-key-123"
                    and self.headers.get("anthropic-version") == "2023-06-01"
                )
                reply = _anthropic_reply(state, body) if ok else {"error": "bad auth/path"}
            else:
                ok = self.path == "/v1/chat/completions" and self.headers.get("Authorization") == "Bearer test-key-123"
                reply = _openai_reply(state, body) if ok else {"error": "bad auth/path"}
            data = json.dumps(reply).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop() -> None:
        server.shutdown()
        thread.join(timeout=5)

    base = f"http://127.0.0.1:{server.server_address[1]}"
    return (base + "/v1") if kind == "openai" else base, state, stop
