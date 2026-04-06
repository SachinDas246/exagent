from .provider import BaseProvider


def get_provider(name: str, **kwargs) -> BaseProvider:
    """Create a provider instance by name."""
    normalized_name = name.strip().lower()

    if normalized_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(**kwargs)

    if normalized_name in {"anthropic", "claude"}:
        from .anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)

    raise ValueError(f"Unsupported provider: {name}")
