# viz/executor.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def extract_code(llm_output) -> str:
    """
    Handles both plain string and structured content block responses
    from different LangChain/Gemini versions.
    """
    # ✅ If it's already a plain string
    if isinstance(llm_output, str):
        code = llm_output.strip()

    # ✅ If it's a list of content blocks (newer Gemini format)
    elif isinstance(llm_output, list):
        code = ""
        for block in llm_output:
            if isinstance(block, dict) and block.get("type") == "text":
                code += block.get("text", "")
            elif isinstance(block, str):
                code += block
        code = code.strip()

    else:
        code = str(llm_output).strip()

    # Clean up markdown fences if present
    code = code.replace("```python", "").replace("```", "").strip()

    return code


def execute_plot(llm_output, df):
    """
    Executes LLM-generated Plotly code and returns (fig, error)
    """
    code = extract_code(llm_output)

    # Reject if LLM returned ERROR
    if not code or code.upper().startswith("ERROR"):
        return None, "LLM returned ERROR"

    try:
        local_vars = {
            "df": df,
            "px": px,
            "go": go,
            "pd": pd
        }
        exec(code, local_vars)

        fig = local_vars.get("fig")
        if fig is None:
            return None, "No figure named `fig` was created"

        return fig, None

    except Exception as e:
        return None, str(e)