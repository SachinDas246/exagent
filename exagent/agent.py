from typing import Any, Callable, Iterator

from .providers import BaseProvider, get_provider
from .model import Model
from .tool import Tool
from .types import ProviderResponse, ToolCall
from .utils import load_skill


class SystemSkillClass:
    def __init__(self):
        if not hasattr(self, "skills"):
            self.skills: list[dict[str, str]] = []

    def add_system_skill(self, skill: dict[str, str]) -> None:
        """Register a skill on the agent."""
        if not hasattr(self, "skills"):
            self.skills = []
        self.skills.append(skill)

    def load_system_skill(self, path: str) -> dict[str, str]:
        """Load and register a skill from disk."""
        skill = load_skill(path)
        self.add_system_skill(skill)
        return skill

    def load_system_skills(self, paths: list[str]) -> list[dict[str, str]]:
        """Load and register multiple skills from disk."""
        for path in paths:
            self.load_system_skill(path)
        return self.skills


class Agent(SystemSkillClass):
    _system_prompt = "You are a helpful assistant."

    def __init__(self):
        super().__init__()
        if not hasattr(self, "tools"):
            self.tools: list[Tool] = []
        if not hasattr(self, "system_description"):
            self.system_description: str | None = None
        if not hasattr(self, "model"):
            self.model: Model | None = None

        self.system_prompt = self.system_description or self._system_prompt

        if self.skills:
            skill_blocks = [
                f"Skill: {skill['name']}\nDescription: {skill['description']}"
                for skill in self.skills
            ]
            self.system_prompt = (
                f"{self.system_prompt.rstrip()}\n\nAvailable skills:\n\n"
                + "\n\n".join(skill_blocks)
            )

        # Seed chat_history with the system prompt as the first message.
        # If the user pre-populated chat_history before calling super().__init__(),
        # respect that override and leave it alone.
        if not hasattr(self, "chat_history"):
            self.chat_history: list[dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt}
            ]

    # -- model config -------------------------------------------------------

    def set_model(self, provider: str, model, **provider_kwargs) -> Model:
        """Configure the single active model for this agent."""
        model_instance = Model(provider, model, **provider_kwargs)
        self.model = model_instance
        return model_instance

    # -- tool registry ------------------------------------------------------

    def add_tool(self, tool: Tool | Callable) -> Tool:
        """Register a tool. Accepts a Tool instance or a plain function
        (which will be wrapped via the `@tool` decorator on the fly)."""
        if not hasattr(self, "tools"):
            self.tools = []
        if not isinstance(tool, Tool):
            from .tool import tool as tool_decorator
            tool = tool_decorator(tool)
        self.tools.append(tool)
        return tool

    def add_tools(self, tools: list[Tool | Callable]) -> list[Tool]:
        return [self.add_tool(t) for t in tools]

    def _find_tool(self, name: str) -> Tool | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Execute a single tool call and return a canonical tool_result block."""
        tool = self._find_tool(call.name)
        if tool is None:
            return {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": f"Error: tool '{call.name}' is not registered.",
                "is_error": True,
            }
        try:
            result = tool.run(call.input)
        except Exception as exc:
            return {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": f"Error running tool '{call.name}': {exc}",
                "is_error": True,
            }
        if not isinstance(result, str):
            try:
                import json
                result = json.dumps(result)
            except (TypeError, ValueError):
                result = str(result)
        return {"type": "tool_result", "tool_use_id": call.id, "content": result}

    # -- generation ---------------------------------------------------------

    def _handle_turn_complete(
        self,
        provider_response: ProviderResponse,
        iteration: int,
        on_iteration: Callable[[int, ProviderResponse], None] | None,
    ) -> list[dict[str, Any]]:
        """Shared post-turn bookkeeping: fire on_iteration, append assistant
        message to history, execute any requested tools, and append the
        resulting tool_result blocks as a user turn.

        Returns the list of tool_result blocks produced this turn (empty if
        the model stopped requesting tools).
        """
        if on_iteration is not None:
            on_iteration(iteration, provider_response)

        if provider_response.assistant_message is not None:
            self.chat_history.append(provider_response.assistant_message)

        if not provider_response.tool_calls:
            return []

        result_blocks = [
            self._execute_tool_call(call) for call in provider_response.tool_calls
        ]
        self.chat_history.append({"role": "user", "content": result_blocks})
        return result_blocks

    def run(
        self,
        prompt: str,
        max_iterations: int = 10,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_iteration: Callable[[int, ProviderResponse], None] | None = None,
    ) -> str:
        """Drive the agent loop and return the final assistant text.

        Uses the provider's non-streaming API — one blocking call per turn.
        Use `stream()` instead if you want live text and events as they arrive.

        Optional hooks for observability:
            on_tool_call(tool_call)          — fired once per tool the model
                                               requests, just before the handler
                                               runs.
            on_iteration(iteration, response)
                                             — fired at the end of each model
                                               turn with the ProviderResponse.
                                               `iteration` is 1-indexed.
        """
        if self.model is None:
            raise ValueError("No model configured. Use set_model(...) first.")

        self.chat_history.append({"role": "user", "content": prompt})

        last_text = ""
        for iteration in range(1, max_iterations + 1):
            response = self.model.generate(
                self.chat_history,
                system=self.system_prompt,
                tools=self.tools or None,
            )

            if on_tool_call is not None:
                for tc in response.tool_calls:
                    on_tool_call(tc)

            last_text = response.text
            result_blocks = self._handle_turn_complete(response, iteration, on_iteration)

            if not result_blocks:
                return last_text

        return last_text

    def stream(
        self,
        prompt: str,
        max_iterations: int = 10,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_iteration: Callable[[int, ProviderResponse], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Drive the agent loop and yield events as they happen.

        Uses the provider's streaming API so `text_delta` events arrive as
        tokens are generated. Use `run()` instead if you just want the final
        text string.

        Event shapes:
            {"type": "text_delta", "text": str}
            {"type": "tool_call", "tool_call": ToolCall}
            {"type": "tool_result", "id": str, "name": str,
             "content": str, "is_error": bool}
            {"type": "done", "text": str}

        Optional hooks for observability:
            on_tool_call(tool_call)          — fired once per tool the model
                                               requests, just before the handler
                                               runs.
            on_iteration(iteration, response)
                                             — fired at the end of each model
                                               turn with the ProviderResponse.
                                               `iteration` is 1-indexed.
        """
        if self.model is None:
            raise ValueError("No model configured. Use set_model(...) first.")

        self.chat_history.append({"role": "user", "content": prompt})

        last_text = ""
        for iteration in range(1, max_iterations + 1):
            provider_response: ProviderResponse | None = None

            for event in self.model.stream(
                self.chat_history,
                system=self.system_prompt,
                tools=self.tools or None,
            ):
                etype = event.get("type")
                if etype == "message_complete":
                    # Internal event — don't forward; use it to drive the loop.
                    provider_response = event["response"]
                    continue
                if etype == "tool_call" and on_tool_call is not None:
                    on_tool_call(event["tool_call"])
                yield event

            if provider_response is None:
                raise RuntimeError("Provider stream did not emit message_complete.")

            last_text = provider_response.text
            result_blocks = self._handle_turn_complete(
                provider_response, iteration, on_iteration
            )

            if not result_blocks:
                yield {"type": "done", "text": last_text}
                return

            # Surface each tool result as its own event for live consumers.
            for call, block in zip(provider_response.tool_calls, result_blocks):
                yield {
                    "type": "tool_result",
                    "id": call.id,
                    "name": call.name,
                    "content": block.get("content", ""),
                    "is_error": bool(block.get("is_error", False)),
                }

        yield {"type": "done", "text": last_text}
