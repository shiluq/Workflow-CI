name: Machine Learning Pipeline CI

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    # 1. Checkout kode dari repository GitHub
    - name: Checkout Code
      uses: actions/checkout@v3

    # 2. Setup Python di server GitHub Virtual Machine
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    # 3. Install semua library utama termasuk MLflow
    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install mlflow scikit-learn pandas joblib

    # 4. MENJALANKAN TRAINING SEBENARNYA (Sesuai Kriteria 3 dari Reviewer)
    - name: Run MLflow Training
      run: |
        # Perintah sakti ini akan mengeksekusi file MLProject kamu secara otomatis di server GitHub
        mlflow run MLProject --no-conda