# Instilit: Global Salary Intelligence ML Pipeline

## Overview

Instilit is an end-to-end machine learning pipeline for global salary benchmarking and compensation prediction. It handles data ingestion, preprocessing, modeling, experiment tracking, deployment, and monitoring.

## Features

- Data ingestion from PostgreSQL
- Data cleaning, EDA, feature engineering
- Multiple model training and selection
- MLflow experiment tracking
- Drift detection with Evidently
- Automated retraining with Airflow
- REST API with Flask
- Dockerized deployment

## Project Structure

```
instilit_ml_pipeline/
├── data/
├── src/
├── airflow/
├── notebooks/
├── tests/
├── docker/
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Configure environment variables** (see `src/config/config.py`)
4. **Run with Docker Compose**
   ```
   docker-compose -f docker/docker-compose.yml up --build
   ```
5. **Access the API**
   - Visit [http://localhost:5000/predict](http://localhost:5000/predict)

## Usage

- Place raw data in `data/raw/`
- Use notebooks for EDA
- Run Airflow for scheduled pipelines
- Monitor drift and retrain as needed

## License

MIT License

## Conda vs Python venv for Data Science/MLOps

- **Conda** is generally better for data science and MLOps projects because:
  - It manages both Python versions and native dependencies (C/C++ libraries) easily.
  - It simplifies installing packages like numpy, pandas, scikit-learn, and others that may require compilation.
  - It allows you to create and manage multiple isolated environments with different Python versions.
  - It is widely used in the data science community and works well with Jupyter, MLflow, Airflow, etc.

- **Python venv** is lightweight and built-in, but:
  - It only manages Python packages, not system-level dependencies.
  - You may face more issues with binary packages or complex dependencies.

**Recommendation:**  
Use **conda** if you want easier environment management and fewer dependency issues, especially for data science/MLOps projects.
Use **venv** if you want a minimal, pure-Python environment and are comfortable managing system dependencies yourself.

## How to activate Conda environments

### 1. Activate your base environment
```bash
conda activate base
```

### 2. Create and activate a dedicated environment for Airflow
```bash
conda create -n airflow-env python=3.10
conda activate airflow-env
```

### 3. (Optional) Create and activate a dedicated environment for MLflow or your main ML pipeline
```bash
conda create -n mlflow-env python=3.10
conda activate mlflow-env
```
