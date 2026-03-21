def build_initial_prompt(metadata):
    prompt = """
You are a data visualization assistant.

A pandas DataFrame named `df` already exists.
DO NOT create a new dataframe.
DO NOT load any CSV or file.
Use ONLY Plotly express or plotly.graph_objects.
"""

    prompt += "\nDataset schema:\n"

    for col in metadata:
        if col["type"] == "categorical":
            # Limit to top 20 values as the paper suggests
            values = col["values"][:20] if len(col["values"]) > 20 else col["values"]
            prompt += f"- {col['name']} (categorical): {values}\n"
        else:
            prompt += f"- {col['name']} ({col['type']})\n"

    prompt += """
Rules:
- Generate Python Plotly code only at any cost
- Store the final figure in variable `fig`
- Add a meaningful title (NOT the user query as title)
- Add axis labels where applicable
- Do NOT use markdown or ``` fences
- Do NOT call fig.show()
- Do NOT import pandas or load any file
- If the query is unrelated to the dataset, return ONLY the word: ERROR
"""

    return prompt