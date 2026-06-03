import pandas as pd
import numpy as np

from config import (
    MAX_SAMPLE_VALUES,
    TOP_VALUE_COUNT
)


def detect_column_type(series):
    """
    Detect generic column type.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

    if unique_ratio < 0.2:
        return "categorical"

    return "text"


def profile_column(series):
    """
    Build profile for a single column.
    """

    total_rows = len(series)

    profile = {
        "column_name": series.name,
        "column_type": detect_column_type(series),
        "dtype": str(series.dtype),
        "total_rows": total_rows,
        "missing_count": int(series.isna().sum()),
        "missing_percent": round(
            (series.isna().sum() / total_rows) * 100,
            2
        ),
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": (
            series.dropna()
            .astype(str)
            .unique()[:MAX_SAMPLE_VALUES]
            .tolist()
        )
    }

    value_counts = (
        series.astype(str)
        .value_counts()
        .head(TOP_VALUE_COUNT)
        .to_dict()
    )

    profile["top_values"] = value_counts

    if pd.api.types.is_numeric_dtype(series):

        profile["min"] = (
            None if series.dropna().empty
            else float(series.min())
        )

        profile["max"] = (
            None if series.dropna().empty
            else float(series.max())
        )

        profile["mean"] = (
            None if series.dropna().empty
            else round(float(series.mean()), 2)
        )

        profile["std"] = (
            None if series.dropna().empty
            else round(float(series.std()), 2)
        )

    return profile


def build_dataset_profile(df):
    """
    Generate dataset-level profile.
    """

    dataset_profile = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": []
    }

    for column in df.columns:

        column_profile = profile_column(df[column])

        dataset_profile["columns"].append(
            column_profile
        )

    return dataset_profile