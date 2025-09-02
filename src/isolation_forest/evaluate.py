import pandas as pd
import os
import joblib
from configs.constants import UNKNOWN

def preprocess_test_df(new_df, label_encoders):
    new_df = new_df.drop(columns=["traceId", "spanId"])
    for col in new_df.select_dtypes(include=["object", "category"]).columns:
        known_labels = set(label_encoders[col].classes_)
        new_df[col] = new_df[col].apply(lambda x: x if x in known_labels else UNKNOWN)
        if col in label_encoders:
            new_df[col] = label_encoders[col].transform(new_df[col].astype(str))
    return new_df

def predict_new_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    test_data_path = os.path.join(BASE_DIR, "../../data/test_data/test_data.csv")
    model_path = os.path.join(BASE_DIR, "../models/isolation_forest_model.pkl")
    encoders_path = os.path.join(BASE_DIR, "../models/isolation_forest_label_encoders.pkl")

    # Load new_df, model and encoders
    new_df = pd.read_csv(test_data_path, keep_default_na=False, na_values=[])
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)

    # Preprocess test data frame
    new_df = preprocess_test_df(new_df, label_encoders)

    print(new_df["duration"])

    # Predict anomaly score
    preds = model.predict(new_df.values)   # -1 = anomaly, 1 = normal
    scores = model.decision_function(new_df.values)  # lower = more anomalous
    
    print("preds", preds)
    print("scores", scores)
