from keras.models import load_model
import os
import pandas as pd
import numpy as np
import joblib
from configs.constants import UNKNOWN

def preprocess_test_df(new_df, label_encoders, scaler):
    new_df = new_df.drop(columns=["traceId", "spanId"])
    for col in new_df.select_dtypes(include=["object", "category"]).columns:
        known_labels = set(label_encoders[col].classes_)
        new_df[col] = new_df[col].apply(lambda x: x if x in known_labels else UNKNOWN)
        if col in label_encoders:
            new_df[col] = label_encoders[col].transform(new_df[col].astype(str))
    
    # Scale numeric data
    samples= scaler.transform(new_df.values)

    print("samples", samples)
    
    return samples

def predict_new_data():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))

  test_data_path = os.path.join(BASE_DIR, "../../data/test_data/test_data.csv")
  model_path = os.path.join(BASE_DIR, "autoencoder_model.keras")
  encoders_path = os.path.join(BASE_DIR, "autoencoder_label_encoders.pkl")
  scaler_path = os.path.join(BASE_DIR, "autoencoder_scaler.pkl")
  
  # Load new_df, model and encoders
  new_df = pd.read_csv(test_data_path, keep_default_na=False, na_values=[])
  label_encoders = joblib.load(encoders_path)
  scaler = joblib.load(scaler_path)
  model = load_model(model_path)

  samples = preprocess_test_df(new_df, label_encoders, scaler)

  # Reconstruct with autoencoder
  reconstructed = model.predict(samples)

  # Calculate reconstruction error (MSE per sample)
  errors = np.mean(np.square(samples - reconstructed), axis=1)

  print("Reconstruction errors:", errors)


