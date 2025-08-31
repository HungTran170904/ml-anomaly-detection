import pandas as pd
import os
import json

def get_service_name(resource):
    for attr in resource.get("attributes", []):
        if attr.get("key") == "service.name":
            value = attr.get("value", {})
            return next(iter(value.values()), None)
    return None

def get_span_attributes(span):
    fields = {}

    for attr in span.get("attributes", []):
        if attr.get("key") == "http.response.status_code":
            value = attr.get("value", {})
            fields["http.status_code"] = next(iter(value.values()), None)
    
    return fields

def extract_parent_span_data(data_set, span):
    fields = {}
    
    parent_id = span.get("parentSpanId")
    if not parent_id:
        return fields
    
    for data in data_set:
        if data["spanId"] == parent_id:
            fields["parent.service_name"] = data["service_name"]
            fields["parent.operation"] = data["operation"]
            fields["parent.kind"] = data["kind"]
            break
    return fields

# def flatten_attributes(span):
#     flat = {}
#     for attr in span.get("attributes", []):
#         key = "attributes." + attr.get("key")
#         value_dict = attr.get("value", {})

#         if "arrayValue" not in value_dict:
#             value = next(iter(value_dict.values()), None)
#             flat[key] = value
    
#     return flat

def preprocess_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "../../data/raw_data/traces.jsonl")
    output_path = os.path.join(BASE_DIR, "../../data/processed/isolation_forest_data.csv")

    # Read JSON lines
    with open(input_path, "r") as f:
        json_objs = [json.loads(line) for line in f]

    data_set = []
    for traces in json_objs:
        for resourceSpan in traces["resourceSpans"]:
            service_name = get_service_name(resourceSpan["resource"])
            for scopeSpan in resourceSpan["scopeSpans"]:
                instrumentation_library = scopeSpan["scope"]["name"]
                for span in scopeSpan["spans"]:
                    row = {
                        "instrumentation_library": instrumentation_library,
                        "service_name": service_name,
                        "traceId": span["traceId"],
                        "spanId": span["spanId"],
                        "operation": span["name"],
                        "kind": span["kind"],
                        "duration": int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]),
                        "span_status": span.get("status", {}).get("code", 1),
                        **get_span_attributes(span),
                        **extract_parent_span_data(data_set, span)
                    }
                    data_set.append(row)

    df = pd.DataFrame(data_set)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Data exported to {output_path}")
    return df