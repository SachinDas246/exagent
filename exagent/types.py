from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A normalized tool-use request emitted by a model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ProviderResponse:
    """Structured result from a provider.generate() call.

    `assistant_message` is the canonical (Anthropic-shaped) assistant turn
    that should be appended to chat_history verbatim. Providers that speak
    a different wire format must translate into this shape before returning.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    stop_reason: str | None = None
