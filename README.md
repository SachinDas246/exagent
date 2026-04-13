# exagent

A small Python library for building LLM agents. Provides a minimal set of building blocks — tools, an agent loop, streaming, and observability hooks — without pulling in a large framework.

Works with **OpenAI** and **Anthropic** models out of the box.

> **Note:** This project is under active development. APIs may change between versions.

**Full documentation:** [sachindas246.github.io/exagent_docs](https://sachindas246.github.io/exagent_docs/)

## Features

- **Tool calling** with a `@tool` decorator that infers JSON schema from type hints
- **Multi-step agent loop** that chains tool calls automatically until the model is done
- **Streaming or non-streaming** — same API, flip a boolean
- **Observability hooks** for inspecting tool calls and per-iteration responses
- **Provider-agnostic** — OpenAI and Anthropic supported with a unified interface
- **Multi-agent orchestration** — route tasks across specialist agents automatically
- **Small surface area** — one `Agent` class, one `@tool` decorator, a handful of events

## Installation

```bash
pip install exagent[openai]      # OpenAI only
pip install exagent[anthropic]   # Anthropic only
pip install exagent[all]         # both providers
```

Requires Python 3.10+.

Set your API key in the environment (or a `.env` file in your working directory):

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

```python
from exagent import Agent, tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Return current weather for a city."""
    # Stand-in for a real API call
    return f"{city}: 18°C, sunny"

class WeatherAgent(Agent):
    def __init__(self):
        self.system_description = "You are a helpful weather assistant."
        self.set_model("openai", "gpt-4.1-mini")
        self.add_tool(get_weather)
        super().__init__()

agent = WeatherAgent()
answer = agent.run("What's the weather in Paris?")
print(answer)
```

That's it. The agent will:

1. Send your prompt to the model along with the tool definition
2. Receive a tool call request
3. Execute the handler
4. Feed the result back to the model
5. Return the final text response

## Defining tools

Any function decorated with `@tool` becomes a tool the model can call. The JSON schema is inferred from type hints; the description comes from the docstring.

```python
from exagent import tool

@tool
def search_products(query: str, max_results: int = 10) -> list:
    """Search the product catalog.

    Returns a list of matching products.
    """
    ...
```

Supported type hints: `str`, `int`, `float`, `bool`, `list[T]`, `dict`, `Optional[T]`, `Union[T, None]`.

Override the name or description if needed:

```python
@tool(name="lookup_price", description="Get the current price of a product by SKU.")
def price(sku: str) -> float:
    ...
```

For tools that need full control over the schema, construct a `Tool` directly:

```python
from exagent import Tool

my_tool = Tool(
    name="custom",
    description="...",
    parameters={"type": "object", "properties": {...}, "required": [...]},
    handler=lambda **kwargs: ...,
)
agent.add_tool(my_tool)
```

## The agent loop

`agent.run(prompt)` drives a loop that handles multi-step tool chains automatically. Each iteration:

1. Sends the full conversation history + registered tools to the model
2. If the model requests tools, executes them and feeds results back as a user turn
3. If the model replies with plain text, returns it as the final answer

The loop stops when the model stops requesting tools, or when `max_iterations` (default `10`) is reached.

### Multi-step chaining

Tools chain naturally because each iteration sees the full history, including previous tool results. Example with three tools that depend on each other:

```python
@tool
def find_user(email: str) -> dict:
    """Look up a user by email."""
    ...

@tool
def list_orders(user_id: str) -> list:
    """List all orders for a user."""
    ...

@tool
def get_order_status(order_id: str) -> dict:
    """Get the current status of an order."""
    ...

agent.add_tools([find_user, list_orders, get_order_status])
result = agent.run("What's the status of alice@example.com's most recent order?")
```

The agent will call `find_user` → `list_orders` (using the user ID from step 1) → `get_order_status` (using the order ID from step 2) → then reply with a natural-language answer. The model decides the chain; the library just shuttles data between steps.

## Two entry points: `run()` and `stream()`

`agent.run(prompt)` uses the provider's non-streaming API — one blocking request per turn — and returns the final text as a string:

```python
text = agent.run("Summarize these docs")
```

`agent.stream(prompt)` uses the provider's streaming API and yields events as they arrive — ideal for showing live progress in a CLI or UI:

```python
for event in agent.stream("Summarize these docs"):
    if event["type"] == "text_delta":
        print(event["text"], end="", flush=True)
    elif event["type"] == "done":
        print()
```

Both methods drive the same tool-calling loop. The only difference is how you consume the output.

### Event types (from `stream()`)

| Event | When | Payload |
|---|---|---|
| `text_delta` | As tokens arrive | `text: str` |
| `tool_call` | When the model finalizes a tool call | `tool_call: ToolCall` |
| `tool_result` | After a tool handler runs | `id, name, content, is_error` |
| `done` | Loop complete | `text: str` (final assistant text) |

## Observability hooks

Both `run()` and `stream()` accept two optional callbacks for visibility into the loop:

```python
def log_tool(tc):
    print(f"→ {tc.name}({tc.input})")

def log_iteration(i, response):
    print(f"[iter {i}] text={response.text[:60]!r} tools={len(response.tool_calls)}")

agent.run(
    "Check inventory for SKU-42 and reorder if below 10.",
    on_tool_call=log_tool,
    on_iteration=log_iteration,
)
```

- `on_tool_call(tool_call)` fires once per tool call, just before the handler runs
- `on_iteration(iteration, response)` fires at the end of each model turn with the full `ProviderResponse` (text, tool_calls, stop_reason). `iteration` is 1-indexed.

Hooks work in both streaming and non-streaming modes.

## Providers

Switch providers via `set_model(provider, model_name)`. Any model string the underlying SDK accepts works:

```python
self.set_model("openai", "gpt-4.1-mini")
self.set_model("openai", "gpt-4.1")
self.set_model("anthropic", "claude-sonnet-4-5")
self.set_model("anthropic", "claude-opus-4-5")
```

Provider-specific keyword arguments pass through to the SDK client:

```python
self.set_model("anthropic", "claude-sonnet-4-5", max_tokens=4096, api_key="...")
```

## Configuration patterns

### Subclassing (for reusable agents)

```python
class SupportAgent(Agent):
    def __init__(self):
        self.system_description = "You are a customer support assistant."
        self.set_model("openai", "gpt-4.1-mini")
        self.add_tools([find_user, list_orders, get_order_status])
        super().__init__()

agent = SupportAgent()
```

### Inline (for one-off use)

```python
agent = Agent()
agent.system_description = "You are a helpful assistant."
agent.set_model("openai", "gpt-4.1-mini")
agent.add_tool(my_tool)
# Re-run post-init to rebuild the system prompt with tools/skills registered:
Agent.__init__(agent)
```

## Skills (optional)

Skills are markdown files with YAML frontmatter that describe named capabilities:

```markdown
---
name: writing
description: Use this skill when writing clearly for developers.
---

Additional body content here.
```

Load them onto an agent:

```python
self.load_system_skill("skills/writing.md")
self.load_system_skills(["skills/writing.md", "skills/debugging.md"])
```

The skill's `name` and `description` are injected into the system prompt as a menu of capabilities the model should apply when relevant.

### Example: agent with a skill

Given a skill file `skills/concise.md`:

```markdown
---
name: concise-replies
description: Use this skill to keep all responses short and to the point.
---

Reply in plain language. Avoid bullet points unless the user asks for a list.
Never repeat the user's question back to them. Keep answers under three sentences.
```

Load it onto an agent:

```python
from exagent import Agent, tool

@tool
def get_quote(topic: str) -> str:
    """Return a well-known quote on the given topic."""
    quotes = {
        "perseverance": "It does not matter how slowly you go as long as you do not stop.",
        "curiosity": "The important thing is not to stop questioning.",
    }
    return quotes.get(topic.lower(), "No quote found for that topic.")

class QuoteAgent(Agent):
    def __init__(self):
        self.system_description = "You are a helpful assistant that shares quotes."
        self.set_model("anthropic", "claude-sonnet-4-5")
        self.add_tool(get_quote)
        self.load_system_skill("skills/concise.md")
        super().__init__()

agent = QuoteAgent()
print(agent.run("Share a quote about curiosity."))
```

The `concise-replies` skill is listed in the system prompt. The model applies it when composing its reply — keeping the response short — without any extra prompt engineering on your part.

## Orchestrator

`OrchestratorAgent` lets you build a router that delegates tasks to specialist agents automatically. Each registered agent is exposed to the orchestrator's LLM as a tool — the model reads their descriptions and calls whichever fits the task. If a task spans multiple specialists, it calls them in sequence.

```python
from exagent import Agent, tool
from exagent.orchestrator import OrchestratorAgent

# --- Specialist agents ---

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result."""
    ...

class CalculatorAgent(Agent):
    def __init__(self):
        self.system_description = "You are a calculator assistant."
        self.set_model("openai", "gpt-4.1-mini")
        self.add_tool(calculate)
        super().__init__()

@tool
def word_count(text: str) -> str:
    """Count the number of words in the given text."""
    return f"{len(text.split())} words"

class TextAgent(Agent):
    def __init__(self):
        self.system_description = "You are a text processing assistant."
        self.set_model("openai", "gpt-4.1-mini")
        self.add_tool(word_count)
        super().__init__()

# --- Orchestrator ---

class MyOrchestrator(OrchestratorAgent):
    def __init__(self):
        self.set_model("openai", "gpt-4.1-mini")
        self.add_agent(
            CalculatorAgent(),
            name="calculator",
            description="Handles arithmetic and numeric calculations.",
        )
        self.add_agent(
            TextAgent(),
            name="text_processor",
            description="Handles word counts, character counts, and text operations.",
        )
        super().__init__()

orch = MyOrchestrator()
print(orch.run("What is 2847 * 193?"))           # → routed to calculator
print(orch.run("How many words in 'hello world'?"))  # → routed to text_processor
```

### How routing works

The orchestrator's system prompt instructs the model to pick the best agent tool for each task. The model uses the `description` you provide at registration time to make that decision — write descriptions that are specific about what the agent can and cannot handle.

If a task needs two specialists, the model calls them in sequence within a single `run()` / `stream()` call:

```
User: "Calculate 15% of 840, then count the characters in the result."

Orchestrator LLM → calls calculator("840 * 0.15")  → "126.0"
               → calls text_processor("126.0")     → "4 characters"
               → replies: "15% of 840 is 126.0, which has 4 characters."
```

### Registering agents

```python
orchestrator.add_agent(agent, name="agent_name", description="...")
```

- `name` — the tool name the LLM uses to call this agent. Must be unique.
- `description` — what this agent handles and when to use it. If omitted, falls back to the agent's `system_description`.

Each sub-agent uses its own already-configured model. The orchestrator does not control which model the specialists use.

### Streaming with the orchestrator

`stream()` works the same as on a regular agent. `tool_call` events show which specialist was chosen, and `tool_result` events carry what it returned:

```python
for event in orch.stream("What is 128 * 64?"):
    if event["type"] == "text_delta":
        print(event["text"], end="", flush=True)
    elif event["type"] == "tool_call":
        print(f"\n[routing to] {event['tool_call'].name}")
    elif event["type"] == "tool_result":
        print(f"[agent result] {event['content']}")
    elif event["type"] == "done":
        print()
```

## API reference

### `Agent`

- `agent.set_model(provider, model, **kwargs)` — configure the LLM
- `agent.add_tool(tool)` / `agent.add_tools([...])` — register tools (accepts `Tool` instances or raw functions)
- `agent.load_system_skill(path)` / `agent.load_system_skills([paths])` — register skills from disk
- `agent.run(prompt, max_iterations=10, on_tool_call=None, on_iteration=None) -> str` — drive the loop with the non-streaming provider API and return the final text
- `agent.stream(prompt, max_iterations=10, on_tool_call=None, on_iteration=None) -> Iterator[dict]` — drive the loop with the streaming provider API and yield events
- `agent.chat_history: list` — full conversation history including tool calls and results

### `OrchestratorAgent`

Extends `Agent`. Import from `exagent.orchestrator`.

- `orchestrator.add_agent(agent, name, description=None)` — register a specialist agent as a callable tool. `description` falls back to the agent's `system_description` if omitted.
- All `Agent` methods (`run`, `stream`, `set_model`, hooks) are inherited and work identically.

### `@tool` decorator

```python
@tool
@tool(name="custom_name", description="custom description")
```

### `Tool`

- `Tool(name, description, parameters, handler)` — manual construction
- `tool.run(arguments: dict)` — invoke with a dict of arguments
- `tool.to_anthropic()` / `tool.to_openai()` — provider-specific tool definitions

### Event shapes

```python
{"type": "text_delta", "text": str}
{"type": "tool_call", "tool_call": ToolCall}
{"type": "tool_result", "id": str, "name": str, "content": str, "is_error": bool}
{"type": "done", "text": str}
```

## License

MIT
