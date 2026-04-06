import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints


_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp: Any) -> dict[str, Any]:
    """Best-effort mapping from a Python type hint to a JSON Schema fragment."""
    if tp is inspect.Parameter.empty or tp is Any:
        return {"type": "string"}

    origin = get_origin(tp)
    if origin is None:
        return {"type": _PY_TO_JSON.get(tp, "string")}

    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        item_schema = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    # Union / Optional: pick the first non-None arg
    args = [a for a in get_args(tp) if a is not type(None)]
    if args:
        return _python_type_to_json_schema(args[0])

    return {"type": "string"}


def _build_schema_from_signature(func: Callable) -> dict[str, Any]:
    """Build a JSON Schema `parameters` object from a function signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        schema = _python_type_to_json_schema(hints.get(name, param.annotation))
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


@dataclass
class Tool:
    """A callable tool the model can invoke.

    Normally built via the `@tool` decorator, but can be constructed directly
    if you want full control over the schema.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def __call__(self, *args, **kwargs):
        return self.handler(*args, **kwargs)

    def run(self, arguments: dict[str, Any]) -> Any:
        """Invoke the handler with a dict of arguments (as the model emits them)."""
        return self.handler(**(arguments or {}))

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable:
    """Decorator that turns a plain Python function into a `Tool`.

    Name defaults to the function name, description to its docstring, and
    the JSON schema is inferred from type hints. Override any of these by
    passing keyword arguments:

        @tool(name="get_weather", description="Look up weather for a city")
        def weather(city: str, units: str = "celsius") -> str:
            ...
    """

    def wrap(func: Callable) -> Tool:
        doc = description if description is not None else (inspect.getdoc(func) or "")
        return Tool(
            name=name or func.__name__,
            description=doc,
            parameters=_build_schema_from_signature(func),
            handler=func,
        )

    if _func is not None and callable(_func):
        return wrap(_func)
    return wrap
