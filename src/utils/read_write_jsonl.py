import pandas as pd
import os

def read_jsonl_file(file_path):
    try:
        return pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def write_jsonl_file(data, file_path):
    try:
        if not os.path.exists(file_path):
            open(file_path, 'w').close()
        data.to_json(file_path, orient='records', lines=True)
    except Exception as e:
        print(f"An error occurred: {e}")