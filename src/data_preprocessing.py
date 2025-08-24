import pandas as pd
import os
import jmespath, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../data/raw_data/traces.jsonl")

# Read JSON lines
with open(file_path, "r") as f:
    data = [json.loads(line) for line in f]

# Wrap into DataFrame
df = pd.json_normalize(
    data,
    record_path=["resourceSpans", "scopeSpans", "spans"],  # go into spans
    meta=[
        ["resourceSpans", "resource", "attributes"],
        ["resourceSpans", "schemaUrl"]
    ],
    sep="."
)


# Select the first row (index 0)
first_row = df.iloc[0]

# Print column and value line by line
for col, val in first_row.items():
    print(f"{col}: {val}")
