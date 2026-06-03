import pandas as pd


def load_data(uploaded_file):
    """
    Load CSV or Excel file.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif filename.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file format. Please upload CSV or XLSX."
        )

    return df
