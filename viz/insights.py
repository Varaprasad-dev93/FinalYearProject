import google.generativeai as genai
import os

def get_gemini_vision_insights(pil_image):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
Analyze the given data visualization and extract insights.
Focus on:
- Overall patterns and trends
- Significant variations or outliers
- Comparisons between groups if applicable
- Any notable observations that can be inferred visually

Return the insights in clear, concise bullet points.
"""

    response = model.generate_content([prompt, pil_image])
    return response.text
