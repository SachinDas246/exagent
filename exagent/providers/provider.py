from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from ..tool import Tool
    from ..types import ProviderResponse


class BaseProvider(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list["Tool"] | None = None,
    ) -> "ProviderResponse":
        """Generate a response from the provider (non-streaming).

        Must return a ProviderResponse whose `assistant_message` is in the
        canonical (Anthropic-shaped) format so it can be appended directly
        to chat_history.
        """
        raise NotImplementedError

    def stream(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list["Tool"] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream a response from the provider.

        Yields events shaped like:
            {"type": "text_delta", "text": "..."}
            {"type": "tool_call", "tool_call": ToolCall}
            {"type": "message_complete", "response": ProviderResponse}

        The final `message_complete` event carries the full aggregated
        ProviderResponse (same shape as `generate()` returns) so the caller
        can append `assistant_message` to history and drive a tool loop.
        """
        raise NotImplementedError("Provider does not implement streaming.")
