# viz/insights.py
import google.generativeai as genai
import os
import base64
import io

def get_gemini_vision_insights(fig, df=None):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ No API key found."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        # ✅ Set chromium path for kaleido on Streamlit Cloud
        import plotly.io as pio
        
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        contents = [
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": img_base64
                }
            },
            """Analyze this chart and provide concise bullet point insights:
- Key trends or patterns
- Notable comparisons between groups
- Any outliers or anomalies
- One-line summary
Do NOT use ### headers. Max 6 bullets."""
        ]

        response = model.generate_content(contents)
        return response.text

    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Try again in a moment."
        return f"⚠️ Insights unavailable: {str(e)}"