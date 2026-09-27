"""File read/write tool, sandboxed to a single workspace directory.

Path traversal (e.g. "../../etc/passwd") is rejected so an LLM-driven
tool call can never escape the workspace.
"""
from __future__ import annotations

import os

from . import Tool, ToolResult
from .. import config


class FileWriteTool(Tool):
    name = "write_file"
    description = "Write text content to a file inside the agent's workspace directory."
    input_schema = {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Relative filename, e.g. 'notes.txt'."},
            "content": {"type": "string"},
        },
        "required": ["filename", "content"],
    }

    def __init__(self, workspace_dir: str = config.WORKSPACE_DIR):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _resolve(self, filename: str) -> str:
        path = os.path.abspath(os.path.join(self.workspace_dir, filename))
        if not path.startswith(self.workspace_dir):
            raise ValueError("Path escapes workspace directory — rejected.")
        return path

    def run(self, filename: str, content: str) -> ToolResult:
        path = self._resolve(filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(ok=True, content=f"Wrote {len(content)} chars to {filename}")


class FileReadTool(Tool):
    name = "read_file"
    description = "Read text content from a file inside the agent's workspace directory."
    input_schema = {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
    }

    def __init__(self, workspace_dir: str = config.WORKSPACE_DIR):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _resolve(self, filename: str) -> str:
        path = os.path.abspath(os.path.join(self.workspace_dir, filename))
        if not path.startswith(self.workspace_dir):
            raise ValueError("Path escapes workspace directory — rejected.")
        return path

    def run(self, filename: str) -> ToolResult:
        path = self._resolve(filename)
        if not os.path.exists(path):
            return ToolResult(ok=False, content="", error=f"{filename} does not exist.")
        with open(path, "r", encoding="utf-8") as f:
            return ToolResult(ok=True, content=f.read())
