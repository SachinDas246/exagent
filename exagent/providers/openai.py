import json
from typing import Any, Iterator

from .utils import resolve_api_key
from .provider import BaseProvider
from ..tool import Tool
from ..types import ProviderResponse, ToolCall


def _canonical_to_openai_input(
    history: list[dict[str, Any]],
    system: str | None,
) -> list[dict[str, Any]]:
    """Translate canonical (Anthropic-shaped) history into OpenAI Responses `input`.

    Canonical shapes we handle:
      - {"role": "user"|"assistant", "content": "text"}
      - {"role": "assistant", "content": [ {"type":"text","text":...},
                                            {"type":"tool_use","id":...,"name":...,"input":{...}} ]}
      - {"role": "user", "content": [ {"type":"tool_result","tool_use_id":...,"content": ...} ]}
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in history:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            # Skip: system is already carried via the top-level `system` kwarg
            # (seeded into history by Agent.__init__). Avoid duplicating.
            continue

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_chunks: list[str] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_chunks.append(block.get("text", ""))
                elif btype == "tool_use":
                    if text_chunks:
                        out.append({"role": "assistant", "content": "".join(text_chunks)})
                        text_chunks = []
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input") or {}),
                        }
                    )
            if text_chunks:
                out.append({"role": "assistant", "content": "".join(text_chunks)})
            continue

        if role == "user":
            for block in content:
                if block.get("type") == "tool_result":
                    result_content = block.get("content")
                    if not isinstance(result_content, str):
                        result_content = json.dumps(result_content)
                    out.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id"),
                            "output": result_content,
                        }
                    )
            continue

    return out


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        env_path: str = ".env",
        parallel_tool_calls: bool | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI support is not installed. Run: pip install exagent[openai]"
            ) from exc

        api_key = api_key or resolve_api_key("OPENAI_API_KEY", env_path)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.parallel_tool_calls = parallel_tool_calls

    # -- request builder ----------------------------------------------------

    def _build_kwargs(
        self,
        history: list[dict[str, Any]],
        system: str | None,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": _canonical_to_openai_input(history, system),
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        return kwargs

    # -- response parser ----------------------------------------------------

    def _parse_final_response(self, response: Any) -> ProviderResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content_blocks: list[dict[str, Any]] = []

        for item in getattr(response, "output", []) or []:
            itype = getattr(item, "type", None)

            if itype == "message":
                for block in getattr(item, "content", []) or []:
                    btype = getattr(block, "type", None)
                    if btype in ("output_text", "text"):
                        text = getattr(block, "text", "") or ""
                        text_parts.append(text)
                        content_blocks.append({"type": "text", "text": text})

            elif itype == "function_call":
                raw_args = getattr(item, "arguments", "") or "{}"
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    parsed = {}
                call_id = getattr(item, "call_id", None) or getattr(item, "id", "")
                name = getattr(item, "name", "")
                tool_calls.append(ToolCall(id=call_id, name=name, input=parsed))
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": parsed,
                    }
                )

        return ProviderResponse(
            text="".join(text_parts) or (getattr(response, "output_text", "") or ""),
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": content_blocks},
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    # -- non-streaming ------------------------------------------------------

    def generate(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        kwargs = self._build_kwargs(history, system, tools)
        response = self.client.responses.create(**kwargs)
        return self._parse_final_response(response)

    # -- streaming ----------------------------------------------------------

    def stream(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        kwargs = self._build_kwargs(history, system, tools)

        with self.client.responses.stream(**kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield {"type": "text_delta", "text": delta}
                # function_call argument deltas and other granular events
                # are intentionally ignored — we surface tool_call only once
                # the arguments are fully assembled.

            final_response = stream.get_final_response()

        response = self._parse_final_response(final_response)

        for tc in response.tool_calls:
            yield {"type": "tool_call", "tool_call": tc}

        yield {"type": "message_complete", "response": response}
