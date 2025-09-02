import pandas as pd
import os
import joblib 
from keras import layers, models
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import  LabelEncoder, MinMaxScaler
from configs.constants import UNKNOWN

# Threshold 1 0.16981121936331584
# Threshold 2 0.09720840512546322

def preprocess_df(df):
    # Drop unnecessary columns (ignore if not present)
    df = df.drop(columns=["traceId", "spanId"], errors="ignore")

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        le.fit(list(df[col].astype(str).unique()) + [UNKNOWN])
        df[col] = le.transform(df[col].astype(str))
        label_encoders[col] = le 
    
    # Scale numeric data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.values)

    print("data set", X_scaled[:5])
    
    return X_scaled, label_encoders, scaler

def print_anomaly_threshold(X_scaled, model):
  train_reconstructions = model.predict(X_scaled)
  train_errors = np.mean(np.square(X_scaled - train_reconstructions), axis=1)

  # Option 1: Mean + 2*StdDev
  threshold_1 = np.mean(train_errors) + 2*np.std(train_errors)
  print("Threshold 1", threshold_1)

  # Option 2: 95th percentile
  threshold_2 = np.percentile(train_errors, 95)
  print("Threshold 2", threshold_2)

def train_model():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  
  # Load data set
  data_set_path = os.path.join(BASE_DIR, "../../data/processed/data_set.csv")
  df = pd.read_csv(data_set_path, keep_default_na=False, na_values=[])
  
  # Preprocess data frame
  X_scaled, label_encoders, scaler = preprocess_df(df)
  
  input_dim = X_scaled.shape[1]   # number of features
  print("input_dim", input_dim)
  encoding_dim = 3          # size of latent space

  # Encoder
  input_layer = layers.Input(shape=(input_dim,))
  encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)

  # Decoder
  decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

  # Autoencoder model
  autoencoder = models.Model(inputs=input_layer, outputs=decoded)
  autoencoder.compile(optimizer="adam", loss="mse")

  # Train model
  early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
  autoencoder.fit(
    X_scaled, X_scaled,   # input = output
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
  )

  # Print anomaly threshold
  print_anomaly_threshold(X_scaled, autoencoder)

  # Save model, label encoder and scaler
  model_path = os.path.join(BASE_DIR, "autoencoder_model.keras")
  encoders_path = os.path.join(BASE_DIR, "autoencoder_label_encoders.pkl")
  scaler_path = os.path.join(BASE_DIR, "autoencoder_scaler.pkl")
  
  autoencoder.save(model_path)
  joblib.dump(label_encoders, encoders_path)
  joblib.dump(scaler, scaler_path)

