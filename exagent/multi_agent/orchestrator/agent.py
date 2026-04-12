from ...agent import Agent
from ...tool import Tool


class OrchestratorAgent(Agent):
    """An agent that routes tasks to registered specialist agents.

    Each specialist is registered via add_agent() and exposed to the
    orchestrator's LLM as a tool. The LLM picks the right agent(s),
    calls them in sequence if needed, and surfaces the final result.

    Usage::

        class MyOrchestrator(OrchestratorAgent):
            def __init__(self):
                self.set_model("openai", "gpt-4.1-mini")
                self.add_agent(my_agent, name="agent_name", description="...")
                super().__init__()
    """

    _system_prompt = (
        "You are an orchestrator. You have a set of specialist agents available as tools. "
        "When given a task, call the most suitable agent tool to handle it. "
        "If a task spans multiple specialists, call them in sequence and combine their results. "
        "Always return the final answer to the user clearly and concisely."
    )

    def add_agent(
        self,
        agent: Agent,
        name: str,
        description: str | None = None,
    ) -> None:
        """Register a specialist agent as a callable tool.

        The agent must already have a model configured before being registered.

        Args:
            agent:       A fully configured Agent instance.
            name:        Tool name the LLM will use to invoke this agent.
                         Must be unique across registered agents.
            description: What this agent does and when to use it. Falls back
                         to the agent's system_description if omitted.
        """
        desc = description or getattr(agent, "system_description", None) or f"Agent: {name}"

        def _handler(task: str) -> str:
            return agent.run(task)

        tool = Tool(
            name=name,
            description=desc,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question to send to this agent.",
                    }
                },
                "required": ["task"],
            },
            handler=_handler,
        )
        self.add_tool(tool)
