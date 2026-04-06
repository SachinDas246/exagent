from typing import Any, Iterator

from .providers import BaseProvider, get_provider
from .tool import Tool
from .types import ProviderResponse


class Model:
    """Single model configuration that can create a provider instance on demand."""

    def __init__(self, provider: str, model, **provider_kwargs):
        self.provider = provider
        self.model = model
        self.provider_kwargs = provider_kwargs
        self._provider_instance: BaseProvider | None = None

    def create_provider(self) -> BaseProvider:
        """Instantiate the configured provider once and reuse it."""
        if self._provider_instance is None:
            self._provider_instance = get_provider(
                self.provider,
                model=self.model,
                **self.provider_kwargs,
            )
        return self._provider_instance

    def generate(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        """Non-streaming generation."""
        return self.create_provider().generate(history, system=system, tools=tools)

    def stream(
        self,
        history: list[dict[str, Any]],
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streaming generation — yields provider events."""
        return self.create_provider().stream(history, system=system, tools=tools)
