import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Fill missing values column-wise
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def extract_metadata(df: pd.DataFrame):
    metadata = []

    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            values = df[col].value_counts().head(20).index.tolist()
            metadata.append({
                "name": col,
                "type": "categorical",
                "values": values
            })
        else:
            metadata.append({
                "name": col,
                "type": str(df[col].dtype)
            })

    return metadata
