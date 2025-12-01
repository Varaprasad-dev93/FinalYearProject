import streamlit as st
import pandas as pd
import google.generativeai as genai

from data_processing import preprocess_dataframe, get_dataframe_schema_text
from llm_visualizer import build_visualization_code
from code_executor import extract_code_from_markdown, execute_plot_code

# --- PAGE CONFIG ---
st.set_page_config(page_title="Visistant", layout="wide")
st.title("🧠 Visistant - Natural Language to Data Visualization")

# --- GEMINI SETUP ---
def get_gemini_api_key() -> str:
    """
    Read Gemini API key from Streamlit secrets or environment variable.

    Put this in .streamlit/secrets.toml:
    [gemini]
    api_key = "YOUR_GEMINI_API_KEY"
    """
    # Prefer Streamlit secrets if available
    if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        return st.secrets["gemini"]["api_key"]

    # Fallback to environment variable (e.g. export GEMINI_API_KEY="...")
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "❌ Gemini API key not found.\n\n"
            "Please set it in `.streamlit/secrets.toml` under `[gemini] api_key` "
            "or as environment variable `GEMINI_API_KEY`."
        )
        st.stop()
    return api_key


# Cache the model so it isn't re-created on every rerun
@st.cache_resource
def load_gemini_model():
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")


model = load_gemini_model()

# --- SIDEBAR ---
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.write(
    """
- Try questions like:
  - `Total sales by region`
  - `Average age by gender`
  - `Trend of revenue over time`
- The app will:
  1. Clean your data
  2. Ask Gemini to generate Plotly code
  3. Run that code and show the chart
"""
)

# --- MAIN APP LOGIC ---
if uploaded_file is None:
    st.info("📤 Upload a CSV file from the sidebar to get started.")
    st.stop()

# Load and clean data
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to read CSV: {e}")
    st.stop()

df_cleaned = preprocess_dataframe(df_raw)

# Show preview
st.subheader("📄 Raw Data (First 5 Rows)")
st.dataframe(df_raw.head())

st.subheader("🧽 Cleaned Data (First 5 Rows)")
st.dataframe(df_cleaned.head())

with st.expander("🧾 Data Cleaning Summary", expanded=False):
    st.write("✅ Duplicates removed")
    st.write("✅ NaN numeric values → filled with column mean")
    st.write("✅ NaN categorical values → filled with column mode")
    st.write("ℹ️ Original data is not modified on disk; cleaning is in-memory only.")

# User query
st.subheader("🎨 Ask for a Visualization")
query = st.text_input(
    "Describe what you want to see (e.g. 'Total sales by region as a bar chart'):"
)

if not query:
    st.stop()

# Build dataset description for the prompt
schema_text = get_dataframe_schema_text(df_cleaned)

if st.button("Generate Visualization"):
    with st.spinner("🧠 Thinking and generating Plotly code..."):
        # Ask Gemini for Plotly visualization code (with retry logic inside)
        code_block, last_error = build_visualization_code(
            model=model,
            df_schema_text=schema_text,
            user_query=query,
            max_attempts=3,
        )

    if code_block is None:
        st.error("❌ Could not generate a valid Plotly visualization after several attempts.")
        if last_error:
            st.error("Last error:\n\n" + last_error)
        st.stop()

    # Extract pure python code from ```python ... ``` block
    python_code = extract_code_from_markdown(code_block)

    st.subheader("🧾 Generated Plotly Code")
    st.code(python_code, language="python")

    # Execute code safely with df_cleaned available as `df`
    try:
        fig = execute_plot_code(python_code, df_cleaned)
        if fig is None:
            st.error("⚠️ The generated code did not create a Plotly figure named `fig`.")
        else:
            st.success("✅ Visualization generated successfully!")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("❌ Error while executing the generated code.")
        st.exception(e)
