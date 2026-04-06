
import os
from pathlib import Path

def load_env_file(path: str = ".env") -> dict[str, str]:
      """Load simple KEY=VALUE pairs from an env file."""
      env_path = Path(path).resolve()
      if not env_path.exists():
          return {}

      values: dict[str, str] = {}
      for raw_line in env_path.read_text(encoding="utf-8").splitlines():
          line = raw_line.strip()
          if not line or line.startswith("#") or "=" not in line:
              continue

          key, value = line.split("=", 1)
          values[key.strip()] = value.strip().strip("'\"")

      return values


def resolve_api_key(env_var_name: str, env_path: str = ".env") -> str:
      """Resolve an API key from process env first, then from a local env file."""
      value = os.getenv(env_var_name)
      if value:
          return value

      env_values = load_env_file(env_path)
      value = env_values.get(env_var_name)
      if value:
          return value

      raise ValueError(
          f"Missing API key '{env_var_name}'. Set it in the environment or in '{Path(env_path).resolve()}'."
      )
