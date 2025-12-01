from typing import Optional, Tuple
import traceback


def build_base_prompt(df_schema_text: str, user_query: str) -> str:
    """
    Construct the main prompt for the LLM using the DataFrame schema
    and the user's natural language query.
    """
    prompt = f"""
You are a senior Python data analyst.

A cleaned pandas DataFrame named `df` is already loaded in memory with the following schema:
{df_schema_text}

Rules:
- The DataFrame is already cleaned: missing values are imputed and duplicates are removed.
- Only use the columns listed above.
- Use Plotly Express (imported as `px`) to create visualizations.
- Always assign the chart to a variable named `fig`.
- Do NOT call `fig.show()`.
- Add a clear and descriptive chart title.
- If aggregation is needed (sum, mean, count, etc.), use pandas groupby operations.

User query:
\"\"\"{user_query}\"\"\"

Now:
1. Decide the best Plotly chart type (bar, line, scatter, pie, box, histogram, etc.).
2. Write ONLY Python code wrapped inside a ```python ... ``` block.
3. The code should assume `df` is already defined.
4. Do not print anything; just build `fig`.
"""
    return prompt.strip()


def build_prompt_with_error(df_schema_text: str, user_query: str, error_message: str) -> str:
    """
    Adjust the prompt after a failed attempt, providing the previous error
    so the LLM can correct its code.
    """
    base_prompt = build_base_prompt(df_schema_text, user_query)
    error_note = f"""

The previous code failed with the following error:

{error_message}

Please carefully fix the issue and regenerate fully-correct Plotly code.
Remember:
- Use only columns that actually exist.
- Ensure all variable names are defined.
- Always return a `fig` object.
"""
    return base_prompt + error_note


def build_visualization_code(
    model,
    df_schema_text: str,
    user_query: str,
    max_attempts: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Interact with the Gemini model to generate Plotly code.
    Retries up to `max_attempts` times, passing previous errors back into the prompt.

    Returns:
        code_block: The raw text response from the model containing code.
        last_error: Full traceback of the last error (if all attempts fail).
    """
    attempt = 0
    last_error: Optional[str] = None
    code_block: Optional[str] = None

    while attempt < max_attempts:
        try:
            if attempt == 0 or last_error is None:
                prompt = build_base_prompt(df_schema_text, user_query)
            else:
                prompt = build_prompt_with_error(df_schema_text, user_query, last_error)

            response = model.generate_content(prompt)
            code_block = response.text

            # If we reach here without exceptions, return the code_block
            return code_block, last_error

        except Exception:
            # Any client-side error (API, network, etc.) gets captured
            last_error = traceback.format_exc()
            attempt += 1

    # All attempts failed
    return None, last_error
