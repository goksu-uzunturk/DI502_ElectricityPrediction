import pandas as pd
import numpy as np
import joblib
import zipfile
import os

def load_filtered_data():
    # Specify the ZIP file name
    zip_filename = "../dataset/filtered.zip"

    # Extract the model file from the ZIP archive
    with zipfile.ZipFile(zip_filename, "r") as archive:
        # Extract the model file (named "your_model.pkl" in this example)
        archive.extract("filtered.pkl")

    # Load the model
    df = joblib.load("filtered.pkl")  # Replace with "pickle.load" if you used pickle

    os.remove("filtered.pkl")

    return df

def break_datetime(df):
    df['timestamp']= pd.to_datetime(df['timestamp'])
    df[['year','weekofyear','dayofweek']]= np.uint16(df['timestamp'].dt.isocalendar())
    df['month']= np.uint8(df['timestamp'].dt.month)
    df['hour']= np.uint8(df['timestamp'].dt.hour)
    return df