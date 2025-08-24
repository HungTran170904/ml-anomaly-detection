import pandas as pd
import os
import jmespath, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../data/raw_data/traces.jsonl")

# Read JSON lines
with open(file_path, "r") as f:
    json_lines = [json.loads(line) for line in f]

# Extrace features
data = []
for json_str in json_lines:
    trace_obj = json.loads(json_str)
    data.append({
        ""
    })




# Select the first row (index 0)
first_row = df.iloc[0]

# Print column and value line by line
for col, val in first_row.items():
    print(f"{col}: {val}")
