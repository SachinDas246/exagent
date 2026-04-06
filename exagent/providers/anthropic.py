from typing import Any, Iterator

from .utils import resolve_api_key
from .provider import BaseProvider
from ..tool import Tool
from ..types import ProviderResponse, ToolCall


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        env_path: str = ".env",
        max_tokens: int = 1024,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic support is not installed. Run: pip install exagent[anthropic]"
            ) from exc

        api_key = api_key or resolve_api_key("ANTHROPIC_API_KEY", env_path)
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    # -- request builder ----------------------------------------------------

    def _build_kwargs(
        self,
        history: list[dict[str, Any]],
        system: str | None,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        # Anthropic requires the system prompt as a top-level parameter,
        # not inside `messages`. Strip any stray system turns defensively.
        messages = [m for m in history if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_anthropic() for t in tools]
        return kwargs

    # -- response parser ----------------------------------------------------

    def _parse_final_message(self, message: Any) -> ProviderResponse:
        """Convert an Anthropic Message into a canonical ProviderResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content_blocks: list[dict[str, Any]] = []

        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                content_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input or {}))
                )
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input or {}),
                    }
                )

        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": content_blocks},
            stop_reason=getattr(message, "stop_reason", None),
        )

    # -- non-streaming ------------------------------------------------------

    def generate(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        kwargs = self._build_kwargs(history, system, tools)
        message = self.client.messages.create(**kwargs)
        return self._parse_final_message(message)

    # -- streaming ----------------------------------------------------------

    def stream(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        kwargs = self._build_kwargs(history, system, tools)

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if getattr(event, "type", None) != "content_block_delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text_delta":
                    yield {"type": "text_delta", "text": delta.text}
                # input_json_delta (streaming tool arguments) is intentionally
                # ignored — we surface the tool_call only once it's complete.

            final_message = stream.get_final_message()

        response = self._parse_final_message(final_message)

        # Emit a tool_call event for each finalized tool use, in order.
        for tc in response.tool_calls:
            yield {"type": "tool_call", "tool_call": tc}

        yield {"type": "message_complete", "response": response}
