import sys

from .agent import Agent

try:
    from .multi_agent.orchestrator.agent import OrchestratorAgent as _OrchestratorAgent
except ImportError:
    _OrchestratorAgent = None


_HELP = """
Commands:
  /clear   — reset the conversation (keeps the system prompt)
  /exit    — quit the shell
  /help    — show this message
"""


def shell(agent: Agent, *, stream: bool = True, prompt: str = "> ") -> None:
    """Start an interactive terminal session with an agent.

    Args:
        agent:  A fully configured Agent (or OrchestratorAgent) instance.
        stream: If True (default), tokens print as they arrive.
                If False, the full response prints after each call returns.
        prompt: The input prompt string. Defaults to '> '.

    Special commands:
        /clear  — reset conversation history (system prompt is preserved)
        /exit   — exit the shell
        /help   — show available commands

    Exit with Ctrl-C or Ctrl-D at any time.
    """
    model_label = getattr(agent, "model", None)
    model_label = getattr(model_label, "model", "unknown model") if model_label else "unknown model"
    mode = "streaming" if stream else "blocking"
    print(f"exagent shell  |  {model_label}  |  {mode}")
    print("Type /help for commands, /exit or Ctrl-C to quit.\n")

    while True:
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            break

        if user_input.lower() == "/help":
            print(_HELP)
            continue

        if user_input.lower() == "/clear":
            # Keep only the system turn at index 0.
            agent.chat_history = agent.chat_history[:1]
            print("Conversation cleared.\n")
            continue

        is_orchestrator = _OrchestratorAgent is not None and isinstance(agent, _OrchestratorAgent)

        if stream:
            _run_streaming(agent, user_input, is_orchestrator=is_orchestrator)
        else:
            _run_blocking(agent, user_input)


def _run_streaming(agent: Agent, user_input: str, *, is_orchestrator: bool = False) -> None:
    try:
        for event in agent.stream(user_input):
            etype = event["type"]

            if etype == "text_delta":
                sys.stdout.write(event["text"])
                sys.stdout.flush()

            elif etype == "tool_call":
                tc = event["tool_call"]
                if is_orchestrator:
                    print(f"\n[delegate] → {tc.name}({tc.input})")
                else:
                    print(f"\n[tool_call] {tc.name}({tc.input})")

            elif etype == "tool_result":
                marker = "error" if event["is_error"] else "ok"
                content = event["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                if is_orchestrator:
                    print(f"[result:{marker}] {event['name']} → {content}")
                else:
                    print(f"[tool_result:{marker}] {event['name']} → {content}")

            elif etype == "done":
                print("\n")

    except Exception as exc:
        print(f"\n[error] {exc}\n")


def _run_blocking(agent: Agent, user_input: str) -> None:
    try:
        result = agent.run(user_input)
        print(f"{result}\n")
    except Exception as exc:
        print(f"[error] {exc}\n")
