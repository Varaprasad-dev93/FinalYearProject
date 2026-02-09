def build_initial_prompt(metadata):
    prompt = """
You are a data visualization assistant.

A pandas DataFrame named `df` already exists.
DO NOT create a new dataframe.
Use ONLY Plotly.
"""

    prompt += "\nDataset schema:\n"

    for col in metadata:
        if col["type"] == "categorical":
            prompt += f"- {col['name']} (categorical): {col['values']}\n"
        else:
            prompt += f"- {col['name']} ({col['type']})\n"

    prompt += """
Rules:
- Generate Python Plotly code only at any cost
- Store output in variable `fig`
- Add title and axis labels
- Do NOT use markdown or ``` fences
- Do NOT call fig.show()
- If you cannot generate code, return ONLY the word: ERROR
"""

    return prompt
