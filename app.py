import streamlit as st
import pandas as pd
import uuid
import os

from data.utils import clean_dataframe, extract_metadata
from llm.prompt_builder import build_initial_prompt
from llm.chain import create_conversation_chain, get_session_history, inject_initial_prompt
from viz.executor import execute_plot
from context.context_id import make_context_id
from viz.insights import get_gemini_vision_insights,config

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
        config()
    else:
        st.warning("Please enter Google API Key")
        st.stop()

# -------------------------------------------------
# SESSION INIT
# -------------------------------------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

if "chat_display" not in st.session_state:
    st.session_state.chat_display = {}

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

# DEBUG = st.sidebar.checkbox("Debug mode")

# if DEBUG:
#     local_path = r"D:\vs studio\customer_churn.csv"
#     if os.path.exists(local_path):
#         class FakeFile:
#             name = "customer_churn.csv"
#             def read(self):
#                 return open(local_path, "rb").read()
#         uploaded_files = [FakeFile()]

if uploaded_files:
    for file in uploaded_files:
        existing_names = [v["name"] for v in st.session_state.datasets.values()]
        if file.name in existing_names:
            continue

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

if not st.session_state.datasets:
    st.info("Upload a CSV to begin.")
    st.stop()

dataset_id = st.selectbox(
    "Select Dataset",
    list(st.session_state.datasets.keys()),
    format_func=lambda x: st.session_state.datasets[x]["name"]
)

dataset = st.session_state.datasets[dataset_id]

if "raw_df" not in dataset:
    st.error("Dataset corrupted. Please re-upload the file.")
    st.stop()

if "clean_df" not in dataset:
    dataset["clean_df"] = clean_dataframe(dataset["raw_df"])

if "contexts" not in dataset:
    dataset["contexts"] = {}

st.sidebar.markdown("### Mode")
advanced_mode = st.sidebar.checkbox("Advanced Mode")
insights_mode = st.sidebar.checkbox("Show Insights", value=True)

data_version = "Cleaned"
if advanced_mode:
    data_version = st.sidebar.radio("Data version", ["Cleaned", "Raw"])

# -------------------------------------------------
# SELECT DATAFRAME
# -------------------------------------------------
df = dataset["clean_df"] if data_version == "Cleaned" else dataset["raw_df"]

# -------------------------------------------------
# ✅ DATASET EXPLORER IN SIDEBAR
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Explorer")

# Summary stats
total_rows, total_cols = df.shape
st.sidebar.markdown(f"**Rows:** {total_rows:,} &nbsp;|&nbsp; **Columns:** {total_cols}")

# Column type breakdown
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

st.sidebar.markdown(
    f"Numeric: **{len(num_cols)}** &nbsp;|&nbsp; "
    f"Categorical: **{len(cat_cols)}** &nbsp;|&nbsp; "
    f"Boolean: **{len(bool_cols)}**"
)

# Per-column detail expander
with st.sidebar.expander("Column Details", expanded=False):
    col_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique = df[col].nunique()

        if dtype in ["object", "category", "bool"]:
            col_type = "Categorical"
        elif "int" in dtype or "float" in dtype:
            col_type = "Numeric"
        else:
            col_type = dtype

        col_data.append({
            "Column": col,
            "Type": col_type,
            "Non-Null": non_null,
            "Nulls": null_count,
            "Unique": unique
        })

    col_df = pd.DataFrame(col_data)
    st.dataframe(col_df, use_container_width=True, hide_index=True)

# Numeric column stats expander
if num_cols:
    with st.sidebar.expander("Numeric Stats", expanded=False):
        st.dataframe(
            df[num_cols].describe().round(2),
            use_container_width=True
        )

# Categorical value counts expander
if cat_cols:
    with st.sidebar.expander("Categorical Counts", expanded=False):
        selected_cat = st.selectbox(
            "Select column",
            cat_cols,
            key="cat_col_selector"
        )
        val_counts = df[selected_cat].value_counts().reset_index()
        val_counts.columns = ["Value", "Count"]
        st.dataframe(val_counts, use_container_width=True, hide_index=True)

# Missing values expander
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    with st.sidebar.expander("Missing Values", expanded=False):
        missing_df = missing.reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df["Missing %"] = ((missing_df["Missing Count"] / total_rows) * 100).round(1)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
else:
    st.sidebar.success("No missing values")

st.sidebar.markdown("---")

if advanced_mode:
    columns = st.sidebar.multiselect("Select columns", df.columns.tolist())
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
# CONTEXT ID
# -------------------------------------------------
context_id = make_context_id(dataset_id, mode, columns)

if context_id not in dataset["contexts"]:
    chain = create_conversation_chain()
    system_prompt = build_initial_prompt(metadata)
    inject_initial_prompt(context_id, system_prompt)
    dataset["contexts"][context_id] = {"chain": chain}
    if context_id not in st.session_state.chat_display:
        st.session_state.chat_display[context_id] = []

chain = dataset["contexts"][context_id]["chain"]

if context_id not in st.session_state.chat_display:
    st.session_state.chat_display[context_id] = []

# -------------------------------------------------
# HELPER
# -------------------------------------------------
def plotly_fig_to_pil(fig):
    try:
        img_bytes = fig.to_image(format="png")
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        st.warning(f"Could not convert chart to image: {str(e)}")
        return None

# -------------------------------------------------
# DISPLAY FULL CHAT HISTORY
# -------------------------------------------------
for entry in st.session_state.chat_display[context_id]:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        if entry["error"]:
            st.error(f"Could not generate visualization: {entry['error']}")
            if DEBUG:
                st.code(entry["llm_output"], language="python")
        else:
            st.plotly_chart(entry["fig"], use_container_width=True)
            if entry.get("insights"):
                st.markdown("### Visual Insights")
                st.write(entry["insights"])

# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------
query = st.chat_input("Ask a question about the data")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Generating visualization..."):
        response = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": context_id}}
        )
        llm_output = response.content
        fig, error = execute_plot(llm_output, df_for_llm)

    insights = None

    with st.chat_message("assistant"):
        if error:
            st.error(f"Could not generate visualization: {error}")
            if DEBUG:
                st.code(llm_output, language="python")
        else:
            st.plotly_chart(fig, use_container_width=True)
            if insights_mode:
                with st.spinner("Generating insights..."):
                    pil_img = plotly_fig_to_pil(fig)
                    if pil_img is not None:
                        insights = get_gemini_vision_insights(pil_img)
                        st.markdown("### Visual Insights")
                        st.write(insights)
                    else:
                        st.info("Insights unavailable — chart could not be converted to image.")

    st.session_state.chat_display[context_id].append({
        "query": query,
        "llm_output": llm_output,
        "fig": fig,
        "error": error,
        "insights": insights
    })