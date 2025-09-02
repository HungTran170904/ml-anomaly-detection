from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import joblib 
from configs.constants import UNKNOWN

def preprocess_df(df):
    df = df.drop(columns=["traceId", "spanId"])

    label_encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        le.fit(list(df[col].astype(str).unique()) + [UNKNOWN])
        df[col] = le.transform(df[col].astype(str))
        label_encoders[col] = le 
    
    return df, label_encoders
    
def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load data set
    data_set_path = os.path.join(BASE_DIR, "../../data/processed/data_set.csv")
    df = pd.read_csv(data_set_path, keep_default_na=False, na_values=[])
    
    # Preprocess data frame
    df, label_encoders = preprocess_df(df)
    print(df.head())

    # Train model
    X = df.values
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    # Export trained model and encoders
    model_path = os.path.join(BASE_DIR, "isolation_forest_model.pkl")
    encoders_path = os.path.join(BASE_DIR, "isolation_forest_label_encoders.pkl")

    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoders_path)


