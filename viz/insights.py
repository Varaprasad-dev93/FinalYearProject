# viz/insights.py
import google.generativeai as genai
import os
import json
import numpy as np

def get_gemini_vision_insights(fig):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return " No API key found."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 1024
        }
    )

    fig_data = fig.to_dict()

    chart_type = ""
    traces_info = []
    for trace in fig_data.get("data", []):
        # ✅ Fix — use len() check instead of truthiness on arrays
        x_data = trace.get("x")
        y_data = trace.get("y")

        trace_info = {
            "type": trace.get("type", "unknown"),
            "name": trace.get("name", ""),
            "x": list(x_data[:20]) if x_data is not None and len(x_data) > 0 else [],
            "y": list(y_data[:20]) if y_data is not None and len(y_data) > 0 else [],
        }
        traces_info.append(trace_info)
        chart_type = trace.get("type", "chart")

    layout = fig_data.get("layout", {})
    title = layout.get("title", {}).get("text", "Untitled Chart")
    xaxis = layout.get("xaxis", {}).get("title", {}).get("text", "")
    yaxis = layout.get("yaxis", {}).get("title", {}).get("text", "")

    chart_summary = f"""
Chart Title: {title}
Chart Type: {chart_type}
X Axis: {xaxis}
Y Axis: {yaxis}
Data traces: {json.dumps(traces_info, default=str)}
"""

    prompt = f"""Based on this chart data, provide insights:

{chart_summary}

Return concise bullet points covering:
- Key trends or patterns
- Notable outliers or comparisons
- One-line summary

Do NOT use ### headers."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "Rate limit reached. Try again in a moment."
        return f"Insights unavailable: {str(e)}"