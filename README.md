# 🌸 Iris Flower Classification Using MLOps

[![CI/CD Pipeline](https://github.com/YashMimrot/iris-flower-classification-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YashMimrot/iris-flower-classification-mlops/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-3.48+-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-0194E2.svg)](https://mlflow.org/)
[![Pytest](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

A comprehensive, production-grade Machine Learning Operations (MLOps) project developed for **Third-Year Engineering (T.Y. AIDS)**. This project demonstrates the entire machine learning lifecycle from reproducible data ingestion and DVC pipeline tracking, through MLflow experiment tracking and automated best-model selection, to a responsive FastAPI REST API and containerized Docker deployment.

---

## 📌 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Key Objectives](#-key-objectives)
3. [Dataset Architecture](#-dataset-architecture)
4. [Machine Learning Algorithms](#-machine-learning-algorithms)
5. [Technology Stack](#-technology-stack)
6. [System Architecture](#-system-architecture)
7. [Repository Structure](#-repository-structure)
8. [Installation & Setup (Windows 11 PowerShell)](#-installation--setup-windows-11-powershell)
9. [Reproducing the MLOps Pipeline with DVC](#-reproducing-the-mlops-pipeline-with-dvc)
10. [Experiment Tracking with MLflow](#-experiment-tracking-with-mlflow)
11. [Running the REST API & Web Dashboard](#-running-the-rest-api--web-dashboard)
12. [Automated Testing with Pytest](#-automated-testing-with-pytest)
13. [Continuous Integration (CI/CD)](#-continuous-integration-cicd)
14. [Docker Containerization](#-docker-containerization)
15. [Empirical Results & Comparison](#-empirical-results--comparison)
16. [Future Scope](#-future-scope)

---

## 📖 Project Overview & Problem Statement
The Iris dataset, originally published by British statistician and biologist Ronald Fisher in 1936, is a canonical multiclass classification problem. The objective is to identify the botanical species of an Iris flower among three distinct classes (*Iris Setosa*, *Iris Versicolor*, and *Iris Virginica*) using four morphological measurements.

While traditional university projects stop at training an offline model in a Jupyter notebook, this project establishes a complete **MLOps engineering pipeline**. It guarantees data versioning, zero data leakage, auditability, metric logging, automated regression testing, API serving, and containerization.

---

## 🎯 Key Objectives
- **Zero Data Leakage:** Ensure strict separation of training ($80\%$) and testing ($20\%$) splits with reproducible random state ($42$) and stratified sampling.
- **Data Version Control:** Use **DVC** to track datasets and orchestrate automated, multi-stage pipelines (`dvc.yaml`).
- **Experiment Tracking:** Use **MLflow** to log hyperparameters, train/test metrics, confusion matrix figures, and model artifacts into `Iris_Classification_Experiment`.
- **Model Comparison & Governance:** Train both a primary **Decision Tree Classifier** and a benchmark **Random Forest Classifier**, objectively selecting the superior model based on weighted F1-score.
- **API & Dashboard:** Serve predictions using **FastAPI** with Pydantic validation schemas, Swagger UI (`/docs`), and an interactive glassmorphism dashboard.
- **Automated Quality Assurance:** Implement 21 unit and integration tests using **Pytest**, executed via **GitHub Actions** CI/CD.
- **Containerization:** Package the complete application into a portable, production **Docker** container.

---

## 🌿 Dataset Architecture
Loaded directly from Scikit-learn's built-in dataset repository:
- **Total Samples:** $150$ flower instances
- **Class Balance:** Perfectly balanced ($50$ samples per class)
- **Features (4 continuous numerical values):**
  1. `sepal_length`: Sepal length in centimeters ($4.3 - 7.9\text{ cm}$)
  2. `sepal_width`: Sepal width in centimeters ($2.0 - 4.4\text{ cm}$)
  3. `petal_length`: Petal length in centimeters ($1.0 - 6.9\text{ cm}$)
  4. `petal_width`: Petal width in centimeters ($0.1 - 2.5\text{ cm}$)
- **Target Classes (3 categories):**
  - `0`: *Iris-setosa* (linearly separable by petal geometry)
  - `1`: *Iris-versicolor* (partially overlapping with Virginica)
  - `2`: *Iris-virginica* (partially overlapping with Versicolor)

---

## 🤖 Machine Learning Algorithms

### 1. Decision Tree Classifier (Primary Model)
- **Mechanism:** Partitions the feature space using recursive binary splitting.
- **Hyperparameters:** `criterion='gini'`, `max_depth=4`, `min_samples_split=2`, `min_samples_leaf=1`, `random_state=42`.
- **Role:** High interpretability baseline model.

### 2. Random Forest Classifier (Comparison Model)
- **Mechanism:** Bagging ensemble of $100$ decorrelated decision trees trained on bootstrap resamples with random feature subsets.
- **Hyperparameters:** `n_estimators=100`, `criterion='gini'`, `max_depth=4`, `min_samples_split=2`, `min_samples_leaf=1`, `random_state=42`.
- **Role:** Variance reduction and superior out-of-sample generalization.

---

## 💻 Technology Stack

| Category | Tools & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11 / 3.12 / 3.13 | Core backend, data processing, and ML algorithms |
| **Data & Scientific** | Pandas, NumPy | Data cleaning, validation, arrays, and schema control |
| **Machine Learning** | Scikit-Learn, Joblib | Model training, stratified splitting, evaluation, serialization |
| **Visualization** | Matplotlib, Seaborn | High-resolution publication EDA and confusion matrices |
| **Data Version Control** | DVC (Data Version Control) | Pipeline orchestration (`dvc.yaml`), dataset hashing |
| **Experiment Tracking** | MLflow | Hyperparameter, metric, and artifact tracking (`mlflow.db`) |
| **API Backend** | FastAPI, Uvicorn, Pydantic | Asynchronous RESTful prediction endpoints and schemas |
| **Testing & CI/CD** | Pytest, GitHub Actions | 21 automated tests and automated GitHub validation |
| **Frontend Dashboard** | HTML5, CSS3, Vanilla JS, Chart.js | Modern glassmorphism UI with live API integration |
| **Containerization** | Docker | Production container image and portable execution |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Data Ingestion & Versioning"]
        A["Scikit-Learn Iris Dataset"] -->|src/data/load_data.py| B["Raw Data (data/raw/iris.csv)"]
        B -->|src/data/validate_data.py| C["Schema & Distribution Validation"]
        C -->|DVC Pipeline Tracking| D["DVC Hash & Cache (.dvc, dvc.yaml)"]
    end

    subgraph Feature_Engineering["2. Preprocessing & Leakage Prevention"]
        D -->|src/features/preprocess.py| E["Stratified Split (80/20, seed=42)"]
        E --> F["Train Partition (120 samples)"]
        E --> G["Test Partition (30 samples)"]
    end

    subgraph Modeling["3. Modeling & Experiment Governance"]
        F & G --> H["Train Decision Tree (max_depth=4)"]
        F & G --> I["Train Random Forest (n=100, max_depth=4)"]
        H & I --> J["MLflow Tracking (sqlite:///mlflow.db)"]
        J --> K["Evaluate: Accuracy, F1, Confusion Matrix"]
        K --> L{"Model Selector: Max F1-Score"}
        L -->|Winner: Random Forest (96.67%)| M["Save models/best_model.joblib"]
        M --> N["Save models/model_metadata.json"]
    end

    subgraph Deployment["4. Serving & Deployment"]
        M & N --> O["FastAPI REST API (src/api/app.py)"]
        O --> P["Interactive Web Dashboard (HTML5/CSS3/Chart.js)"]
        O --> Q["Swagger API Docs (/docs)"]
        O --> R["Docker Container (Port 8000)"]
    end

    subgraph Quality["5. Automated CI/CD Testing"]
        S["Pytest Suite (21 Tests)"] --> T["GitHub Actions CI/CD (.github/workflows/ci-cd.yml)"]
    end
```

---

## 📂 Repository Structure

```
d:\iris_mlops_friend\
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD automated workflow
├── config/
│   ├── __init__.py
│   └── config.py                 # Cross-platform pathlib configuration & schemas
├── data/
│   ├── raw/
│   │   └── iris.csv              # Standardized raw Iris dataset (150 rows)
│   └── processed/
│       ├── train.csv             # Stratified training split (120 rows)
│       └── test.csv              # Stratified testing split (30 rows)
├── models/
│   ├── best_model.joblib         # Production serialized winner (Random Forest)
│   ├── decision_tree.joblib      # Serialized Decision Tree model
│   ├── random_forest.joblib      # Serialized Random Forest model
│   └── model_metadata.json       # Version, timestamp, features, and metrics
├── reports/
│   ├── figures/                  # Publication charts & confusion matrices
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_comparison.png
│   │   ├── dt_confusion_matrix.png
│   │   └── rf_confusion_matrix.png
│   └── metrics.json              # Machine-readable evaluation report
├── src/
│   ├── data/
│   │   ├── load_data.py          # Stage 1: Load and standardize raw dataset
│   │   └── validate_data.py      # Stage 2: Validate integrity & zero nulls
│   ├── eda/
│   │   └── generate_eda.py       # Stage 3: Generate 8 EDA figures
│   ├── features/
│   │   └── preprocess.py         # Stage 4: Feature/target stratified splitting
│   ├── models/
│   │   ├── evaluate.py           # Evaluation metrics & confusion matrix plot
│   │   └── train.py              # Stage 5: Train, track with MLflow, select best
│   └── api/
│       ├── app.py                # FastAPI REST API & static file serving
│       ├── schemas.py            # Pydantic schemas for request validation
│       └── static/               # Interactive frontend web dashboard
│           ├── css/styles.css    # Responsive dark/light theme stylesheet
│           ├── js/app.js         # Frontend controller and Chart.js logic
│           └── index.html        # Complete 7-section single-page dashboard
├── tests/
│   ├── test_dataset.py           # 6 tests for dataset schema & balance
│   ├── test_model.py             # 5 tests for training, loading, & inference
│   └── test_api.py               # 10 tests for FastAPI endpoints & validation
├── params.yaml                   # Central pipeline parameters
├── dvc.yaml                      # DVC pipeline specification
├── dvc.lock                      # DVC execution hash record
├── requirements.txt              # Production dependency specifications
├── Dockerfile                    # Container definition
├── .dockerignore                 # Container file exclusion rules
├── .gitignore                    # Git version control rules
├── README.md                     # Comprehensive technical documentation
├── PROJECT_REPORT.md             # Complete academic capstone project report
├── VIVA_QUESTIONS.md             # 30 viva questions with concise answers
├── PRESENTATION_NOTES.md         # Viva & presentation demo script
├── SETUP_GUIDE.md                # Windows 11 step-by-step setup guide
└── ARCHITECTURE.md               # Detailed architectural deep dive
```

---

## 💻 Installation & Setup (Windows 11 PowerShell)

### Step 1: Open PowerShell in Project Directory
```powershell
cd d:\iris_mlops_friend
```

### Step 2: Configure Process-Scoped Execution Policy
If PowerShell displays `Script Execution Disabled`, allow execution for the current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Reproducing the MLOps Pipeline with DVC

DVC tracks pipeline dependencies, parameters, and outputs. Run the complete end-to-end pipeline with:

```powershell
dvc repro
```

Check the pipeline status:
```powershell
dvc status
```
*Output:* `Data and pipelines are up to date.`

### What is the difference between Local DVC and Remote DVC?
- **Local DVC Tracking:** Computes unique cryptographic content hashes (md5) of files and pipelines, recording state in `dvc.lock` and `.dvc/cache` on your local SSD.
- **Remote DVC Storage:** Synchronizes local cached datasets with cloud object stores (e.g. AWS S3, Google Cloud Storage, Azure Blob). If no remote is configured, files remain safely tracked locally without external cloud dependencies.

---

## 🔬 Experiment Tracking with MLflow

All training runs, hyperparameters, train/test metrics, and confusion matrix artifacts are tracked in the SQLite database `sqlite:///mlflow.db`.

### Launch the MLflow Web Dashboard:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open your browser to: **`http://localhost:5000`**

In MLflow, examine:
- Experiment: **`Iris_Classification_Experiment`**
- Runs: **`Decision_Tree_Run`** and **`Random_Forest_Run`**
- Artifacts: Confusion matrix plots and serialized Scikit-learn model files.

---

## 🚀 Running the REST API & Web Dashboard

Start the FastAPI ASGI server:
```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- **Interactive Dashboard:** Open **`http://localhost:8000`**
- **Swagger Interactive API Documentation:** Open **`http://localhost:8000/docs`**
- **ReDoc Documentation:** Open **`http://localhost:8000/redoc`**
- **System Health Check:** Open **`http://localhost:8000/health`**

### Sample API Prediction via PowerShell (`curl.exe`):
```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'
```
*Response JSON:*
```json
{
  "predicted_class": 0,
  "predicted_species": "Iris-setosa",
  "probabilities": {
    "Iris-setosa": 0.985,
    "Iris-versicolor": 0.015,
    "Iris-virginica": 0.0
  },
  "model_name": "Random Forest Classifier",
  "status": "success"
}
```

---

## 🧪 Automated Testing with Pytest

Execute the automated test suite covering dataset schemas, machine learning models, and API endpoints:

```powershell
pytest -v tests/
```

### Test Coverage Breakdown (21 Tests):
- **`tests/test_dataset.py` (6 tests):** Validates raw dataset existence, 150 rows, zero null values, positive numeric dimensions, and 50-per-class balance.
- **`tests/test_model.py` (5 tests):** Validates joblib artifact presence, model deserialization, valid class outputs (0, 1, 2), probability sum ($=1.0$), and Setosa accuracy.
- **`tests/test_api.py` (10 tests):** Tests root route (`/`), `/health`, `/model-info`, `/predict`, `/predict-batch`, `/api/metrics`, `/api/dataset-stats`, and HTTP 422 error handling for negative values, missing keys, and non-numeric types.

---

## ⚙️ Continuous Integration (CI/CD)

The GitHub Actions configuration (`.github/workflows/ci-cd.yml`) runs on every `push` and `pull_request` to `main`:
1. Checks out repository code.
2. Sets up Python 3.11.
3. Installs dependencies from `requirements.txt`.
4. Executes the full pipeline (`load_data.py`, `validate_data.py`, `generate_eda.py`, `preprocess.py`, `train.py`).
5. Executes `pytest -v tests/` and reports test pass/fail status.

---

## 🐳 Docker Containerization

### Build the Docker Image:
```powershell
docker build -t iris-mlops-app:latest .
```

### Run the Container:
```powershell
docker run -d -p 8000:8000 --name iris-mlops-container iris-mlops-app:latest
```

### Check Container Status:
```powershell
docker ps
```

### Stop the Container:
```powershell
docker stop iris-mlops-container
```

---

## 📊 Empirical Results & Comparison

Evaluated on the exact stratified holdout test split ($n=30$ samples, $10$ per class):

| Metric | Decision Tree Classifier | Random Forest Classifier | Production Decision |
| :--- | :---: | :---: | :---: |
| **Training Accuracy** | $99.17\%$ | $100.0\%$ | Baseline vs Bagging Ensemble |
| **Testing Accuracy** | $93.33\%$ | **$96.67\%$** | $+3.34\%$ Improvement |
| **Weighted Precision** | $93.33\%$ | **$96.97\%$** | High class purity |
| **Weighted Recall** | $93.33\%$ | **$96.67\%$** | Minimal false negatives |
| **Weighted F1-Score** | $93.33\%$ | **$96.66\%$** | **Winner Selected** |
| **Holdout Misclassifications** | $2$ / $30$ | **$1$ / $30$** | Only $1$ Versicolor misclassified as Virginica |

---

## 🔮 Future Scope
1. **Continuous Retraining (CT):** Integrate Evidently AI to detect feature drift and trigger automated model updates.
2. **Cloud DVC Remote:** Configure an Amazon S3 or Google Cloud Storage bucket for enterprise dataset sharing.
3. **Kubernetes Deployment:** Deploy container onto Kubernetes (Minikube or AKS) with autoscaling.
4. **Monitoring:** Prometheus & Grafana telemetry for real-time inference latency and request volume.

---

## 👨‍💻 Project Developer & Academic Attributions
- **Student Developer:** Yash Mimrot
- **Branch:** Third-Year Artificial Intelligence & Data Science Engineering (T.Y. AIDS)
- **Institution:** Government College of Engineering, Kolhapur
