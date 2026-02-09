import plotly.express as px
import plotly.graph_objects as go

def execute_plot(code: str, df):
    # Basic sanity checks
    if not isinstance(code, str):
        return None, "LLM output is not text"

    if "fig" not in code:
        return None, "No Plotly figure found in LLM output"

    # Hard reject obvious natural language
    forbidden_phrases = [
        "I'd be happy",
        "Sure!",
        "Here is",
        "I can help",
        "Let me know"
    ]

    if any(p.lower() in code.lower() for p in forbidden_phrases):
        return None, "LLM returned natural language instead of code"

    local_env = {
        "df": df,
        "px": px,
        "go": go
    }

    try:
        exec(code, {}, local_env)
    except Exception as e:
        return None, f"Code execution failed: {e}"

    fig = local_env.get("fig")

    if fig is None:
        return None, "Figure variable `fig` not created"

    return fig, None
