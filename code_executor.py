import re
from typing import Any

import plotly.graph_objects as go  # Just to ensure Plotly is available
import plotly.express as px        # LLM will use px
import pandas as pd


def extract_code_from_markdown(code_block: str) -> str:
    """
    Extract Python code from a markdown string that may contain ```python ... ``` blocks.
    If no fenced block is found, return the original text.
    """
    if not code_block:
        return ""

    match = re.search(r"```(?:python)?\n(.*?)```", code_block, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return code_block.strip()


def execute_plot_code(code: str, df: pd.DataFrame) -> Any:
    """
    Execute the generated Plotly code in a restricted namespace.

    - Exposes only `df`, `px`, and `go` to the code.
    - Expects the code to create a variable named `fig` (a Plotly Figure).

    Returns:
        fig: Plotly figure object created by the code.

    Raises:
        Any exception that occurs during code execution.
    """
    # Restricted globals: no builtins to reduce risk
    restricted_globals = {
        "__builtins__": {},  # Empty dict removes access to built-ins like open(), etc.
        "px": px,
        "go": go,
        "df": df,
    }

    local_vars = {}

    # WARNING: exec is inherently powerful. This is still not a perfect sandbox,
    # but for a controlled academic project with trusted input it is acceptable.
    exec(code, restricted_globals, local_vars)

    fig = local_vars.get("fig") or restricted_globals.get("fig")
    return fig
