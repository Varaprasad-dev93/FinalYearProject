import streamlit as st
import pandas as pd
import uuid
import os

from data.utils import clean_dataframe, extract_metadata
from llm.prompt_builder import build_initial_prompt
from llm.chain import create_conversation_chain
from viz.executor import execute_plot
from context.context_id import make_context_id
from viz.insights import get_gemini_vision_insights

from PIL import Image
import io
# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Visistant – NL to Visualization")

# -------------------------------------------------
# API KEY HANDLING
# -------------------------------------------------
if "GOOGLE_API_KEY" not in os.environ:
    key = st.sidebar.text_input("Google API Key", type="password")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
    else:
        st.warning("Please enter Google API Key")
        st.stop()

# -------------------------------------------------
# SESSION INIT
# -------------------------------------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
DEBUG = st.sidebar.checkbox("Debug mode")

if DEBUG:
    local_path = "D:\vs studio\customer_churn.csv"  # put a CSV here
    if os.path.exists(local_path):
        uploaded_files = [open(local_path, "rb")]

if uploaded_files:
    for file in uploaded_files:
        dataset_id = str(uuid.uuid4())

        raw_df = pd.read_csv(file)
        clean_df = clean_dataframe(raw_df)

        st.session_state.datasets[dataset_id] = {
            "name": file.name,
            "raw_df": raw_df,
            "clean_df": clean_df,
            "metadata_all": extract_metadata(clean_df),
            "contexts": {}
        }

# -------------------------------------------------
# DATASET SELECTION
# -------------------------------------------------
if not st.session_state.datasets:
    st.info("Upload a CSV to begin.")
    st.stop()

dataset_id = st.selectbox(
    "Select Dataset",
    list(st.session_state.datasets.keys()),
    format_func=lambda x: st.session_state.datasets[x]["name"]
)

dataset = st.session_state.datasets[dataset_id]

# -------------------------------------------------
# BACKWARD-SAFE NORMALIZATION
# -------------------------------------------------
if "raw_df" not in dataset:
    st.error("Dataset corrupted. Please re-upload the file.")
    st.stop()

if "clean_df" not in dataset:
    dataset["clean_df"] = clean_dataframe(dataset["raw_df"])

if "contexts" not in dataset:
    dataset["contexts"] = {}

# -------------------------------------------------
# MODE SELECTION
# -------------------------------------------------
st.sidebar.markdown("### Mode")
advanced_mode = st.sidebar.checkbox("Advanced Mode")

if advanced_mode:
    data_version = st.sidebar.radio("Data version", ["Cleaned", "Raw"])
else:
    data_version = "Cleaned"

# -------------------------------------------------
# SELECT DATAFRAME
# -------------------------------------------------
df = dataset["clean_df"] if data_version == "Cleaned" else dataset["raw_df"]

# -------------------------------------------------
# DEFAULT vs ADVANCED MODE LOGIC
# -------------------------------------------------
if advanced_mode:
    columns = st.sidebar.multiselect(
        "Select columns",
        df.columns.tolist()
    )

    if not columns:
        st.warning("Select at least one column")
        st.stop()

    df_for_llm = df[columns]
    metadata = extract_metadata(df_for_llm)
    mode = "advanced"

else:
    df_for_llm = df
    metadata = dataset["metadata_all"]
    columns = None
    mode = "default"

# -------------------------------------------------
# CONTEXT ID (DATASET × MODE × COLUMNS)
# -------------------------------------------------
context_id = make_context_id(dataset_id, mode, columns)

if context_id not in dataset["contexts"]:
    chain = create_conversation_chain()
    system_prompt = build_initial_prompt(metadata)
    chain.run(system_prompt)

    dataset["contexts"][context_id] = {
        "chain": chain
    }

chain = dataset["contexts"][context_id]["chain"]


# -------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------
query = st.chat_input("Ask a question about the data")

def plotly_fig_to_pil(fig):
    img_bytes = fig.to_image(format="png")
    return Image.open(io.BytesIO(img_bytes))

if query:
    with st.spinner("Generating visualization..."):
        if "last_response" not in st.session_state:
                st.session_state.last_response = None

        st.session_state.last_response = chain.invoke(query)
        response = st.session_state.last_response['response']
        fig, error = execute_plot(response, df_for_llm)
    
    if error:
        st.error("Could not generate visualization.")
        st.code(response)  # show raw LLM output for debugging
    else:
        pil_img = plotly_fig_to_pil(fig)
        insights = get_gemini_vision_insights(pil_img)

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🔍 Visual Insights")
        st.write(insights)

