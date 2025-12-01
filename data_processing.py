import pandas as pd


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame:
    - Drop duplicate rows
    - Fill numeric NaNs with column mean
    - Fill categorical NaNs with column mode (or 'Unknown' if no mode)
    """
    df_clean = df.copy()

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Fill numeric NaNs with mean
    numeric_cols = df_clean.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # Fill categorical NaNs with mode
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df_clean[col].isnull().any():
            mode = df_clean[col].mode()
            df_clean[col] = df_clean[col].fillna(mode[0] if not mode.empty else "Unknown")

    return df_clean


def get_dataframe_schema_text(df: pd.DataFrame, max_unique_for_example: int = 5) -> str:
    """
    Build a textual description of the DataFrame schema to send to the LLM.
    This helps the model understand column names and types.

    Example output:
    - age (int64) | numeric | min=18, max=70
    - gender (object) | categorical | examples: ['Male', 'Female']
    """
    lines = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
            desc = f"min={series.min()}, max={series.max()}"
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
            desc = f"min={series.min()}, max={series.max()}"
        else:
            col_type = "categorical/text"
            unique_vals = series.dropna().unique()
            examples = unique_vals[:max_unique_for_example]
            desc = f"examples={list(map(str, examples))}"

        lines.append(f"- {col} ({dtype}) | {col_type} | {desc}")

    return "\n".join(lines)
