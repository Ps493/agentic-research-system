"""A sandboxed arithmetic tool.

Uses an AST whitelist rather than eval()/exec() so the agent can never
be tricked (by its own LLM output or by adversarial content pulled in
from a web search) into executing arbitrary code.
"""
from __future__ import annotations

import ast
import operator

from . import Tool, ToolResult

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed or invalid expression element: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (+, -, *, /, %, **). Use for any numeric computation."
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '(120 + 45) * 3'"},
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return ToolResult(ok=True, content=str(result))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, content="", error=f"Could not evaluate '{expression}': {exc}")
