import pandas as pd
import os
from typing import Union, Any


def read_jsonl_file(file_path: str) -> Union[pd.DataFrame, None]:
    """
    Reads a JSONL file and returns a pandas DataFrame.

    Args:
        file_path: The path to the JSONL file.

    Returns:
        A pandas DataFrame containing the data from the JSONL file, or None if an error occurred.
    """
    try:
        return pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def write_jsonl_file(data: pd.DataFrame, file_path: str) -> None:
    """
    Writes a pandas DataFrame to a JSONL file.

    Args:
        data: The pandas DataFrame to write.
        file_path: The path to the JSONL file.

    Returns:
        None
    """
    try:
        if not os.path.exists(file_path):
            open(file_path, "w").close()
        data.to_json(file_path, orient="records", lines=True)
    except Exception as e:
        print(f"An error occurred: {e}")
