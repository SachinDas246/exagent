import os
from pathlib import Path

def load_file_as_string(path: str) -> str:
      """Load a file from the given path."""
      with open(Path(path).resolve(), "r", encoding="utf-8") as f:
          return f.read()


def parse_skill_file(path: str) -> dict[str, str]:
      """Parse a markdown skill file with YAML-like frontmatter."""
      content = load_file_as_string(path).strip()

      if not content.startswith("---"):
          raise ValueError("Skill file must start with frontmatter delimited by ---")

      parts = content.split("---", 2)
      if len(parts) < 3:
          raise ValueError("Skill file frontmatter is incomplete.")

      frontmatter = parts[1].strip()
      body = parts[2].strip()

      metadata: dict[str, str] = {}
      for line in frontmatter.splitlines():
          if ":" not in line:
              continue
          key, value = line.split(":", 1)
          metadata[key.strip()] = value.strip()

      if "name" not in metadata or "description" not in metadata:
          raise ValueError("Skill frontmatter must include 'name' and 'description'.")

      metadata["content"] = body
      return metadata


def load_skill(path: str) -> dict[str, str]:
      """Load a single skill file."""
      return parse_skill_file(path)

