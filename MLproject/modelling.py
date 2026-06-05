import os
# KUNCI UTAMA: Paksa MLflow mengizinkan penyimpanan berbasis folder biasa di Windows
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib 

def train_saham_basic():
    # 1. FIX: Gunakan path relatif murni agar sinkron dengan terminal saat MLflow UI dinyalakan
    mlflow.set_tracking_uri("file:mlruns")
    
    # 2. FIX: Set eksperimen duluan sebelum autolog berjalan
    mlflow.set_experiment("Eksperimen_Saham_ANTM_Basic")
    
    # 3. Aktifkan autolog
    mlflow.autolog()

    print("--- 1. Membaca Data Bersih ANTM ---")
    df = pd.read_csv("namadataset_preprocessing.csv")
    print(f"Berhasil memuat data! Total: {len(df)} baris.")

    # Membuat fitur lag untuk data saham
    df['close_lag1'] = df['close'].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df[['close_lag1', 'open', 'high', 'low', 'volume']]
    y = df['close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("\n--- 2. Memulai Training Model dengan Autolog & Manual Log ---")
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        print(f"\nSukses! Model dilatih dengan R2 Score: {r2:.4f}")
        
        # TAMBAHAN KUNCI LULUS: Paksa log model manual ke dalam artefak run ini
        print("Menyimpan folder model ke Artefak MLflow...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        
        # Simpan model.pkl di luar untuk bahan serving Kriteria 4 nanti
        joblib.dump(model, "../model.pkl")
        print("Folder 'mlruns' dan file 'model.pkl' lokal siap digunakan!")

if __name__ == '__main__':
    train_saham_basic()
