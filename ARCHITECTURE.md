# 🏛️ SYSTEM ARCHITECTURE & DATA FLOW SPECIFICATION
## Iris Flower Classification Using MLOps
**Degree:** B.Tech Artificial Intelligence and Data Science (T.Y. AIDS)  
**Author:** Nikhil Landage  
**Institution:** Government College of Engineering, Kolhapur  

---

## 1. High-Level Architectural Diagram

The diagram below visualizes the end-to-end data flow, feedback loops, and boundaries of the Iris MLOps system:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Ingestion & Validation"]
        RawSource["Scikit-Learn Iris Source"] -->|src/data/load_data.py| RawCSV["Raw Data (data/raw/iris.csv)"]
        RawCSV -->|src/data/validate_data.py| Validator{"Schema & Quality Gate"}
        Validator -->|Pass (150 rows, 0 nulls)| ValidData["Validated Dataset"]
        Validator -->|Fail| Abort["Raise DataValidationError"]
    end

    subgraph S2["Stage 2: Versioning & Preprocessing"]
        ValidData -->|Tracked by DVC| DVCLock["DVC Hashing (dvc.yaml / dvc.lock)"]
        DVCLock -->|src/features/preprocess.py| Splitter["Stratified Split (80/20, seed=42)"]
        Splitter --> TrainData["data/processed/train.csv (120 rows)"]
        Splitter --> TestData["data/processed/test.csv (30 rows)"]
    end

    subgraph S3["Stage 3: Exploration & Analysis"]
        RawCSV -->|src/eda/generate_eda.py| EDA["EDA Engine"]
        EDA --> Figures["reports/figures/ (8 PNG Artifacts)"]
    end

    subgraph S4["Stage 4: Training & Experiment Tracking"]
        TrainData & TestData --> DT["Train Decision Tree (max_depth=4)"]
        TrainData & TestData --> RF["Train Random Forest (n=100)"]
        DT & RF --> MLflowStore["MLflow Tracking Store (sqlite:///mlflow.db)"]
        MLflowStore --> MetricsJSON["reports/metrics.json"]
        DT & RF --> Selector{"F1-Score Selector"}
        Selector -->|Winner: RF (96.67%)| BestModel["models/best_model.joblib"]
        Selector --> Metadata["models/model_metadata.json"]
    end

    subgraph S5["Stage 5: Serving & User Interface"]
        BestModel & Metadata --> APIServer["FastAPI Backend (src/api/app.py)"]
        APIServer --> REST["REST Endpoints (/health, /predict, /predict-batch)"]
        APIServer --> UI["Interactive Glassmorphism Dashboard (/)"]
        APIServer --> SwaggerUI["Swagger API Documentation (/docs)"]
    end

    subgraph S6["Stage 6: Quality Assurance & Deployment"]
        APIServer & BestModel --> PytestHarness["Pytest Suite (21 Tests)"]
        PytestHarness --> CI["GitHub Actions CI/CD (.github/workflows/ci-cd.yml)"]
        APIServer & BestModel --> Dockerfile["Docker Container (Port 8000)"]
    end
```

---

## 2. Component-by-Component Architectural Breakdown

### 2.1. Ingestion & Quality Gate (Data Pipeline)
- **Component:** `src/data/load_data.py` and `src/data/validate_data.py`
- **Responsibility:** Ingest raw flower measurements, map column names to standardized snake_case format (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `target`, `species_name`), and verify complete schema integrity before any downstream stage executes.
- **Why it matters:** In real-world MLOps, silent data corruption or unexpected missing values cause severe model drift or inference crashes. The automated quality gate halts the pipeline if invalid dimensions or nulls appear.

### 2.2. Pipeline Orchestration & Data Version Control (DVC)
- **Component:** `dvc.yaml` and `params.yaml`
- **Responsibility:** Define clear input dependencies (`deps`), configuration parameters (`params`), and generated outputs (`outs`) for each stage.
- **Why it matters:** Standard Git cannot version large datasets without repository bloat. DVC computes MD5 content hashes of data files and scripts, storing hashes in `dvc.lock`. When `dvc repro` is called, DVC checks if hashes match, skipping redundant recalculations.

### 2.3. Preprocessing & Leakage Prevention
- **Component:** `src/features/preprocess.py`
- **Responsibility:** Separates features from target labels and applies stratified train-test splitting (80% train = 120 samples, 20% test = 30 samples).
- **Why it matters:** Data leakage occurs when test data characteristics inadvertently leak into training steps. Enforcing a strict, early split with a fixed random seed (`42`) guarantees that evaluation metrics represent true out-of-sample generalization.

### 2.4. Machine Learning Engine & Experiment Governance (MLflow)
- **Component:** `src/models/train.py` and `src/models/evaluate.py`
- **Responsibility:** Trains Decision Tree and Random Forest classifiers, evaluates performance using multiclass weighted and macro metrics, and logs runs, parameters, metrics, confusion matrix plots, and model artifacts to MLflow experiment `Iris_Classification_Experiment` backed by SQLite.
- **Why it matters:** Guarantees transparency and auditability. Anyone can trace which hyperparameter set produced which specific metric score and inspect the exact confusion matrix artifact.

### 2.5. Automated Model Selector & Serialization
- **Component:** `src/models/train.py`
- **Responsibility:** Evaluates test partition weighted F1-scores. The model with superior generalization is automatically chosen as `best_model.joblib`, and comprehensive metadata (timestamp, algorithm, hyperparameters, metrics) is exported to `models/model_metadata.json`.
- **Why it matters:** Removes human guesswork when choosing which model version to deploy to production.

### 2.6. Application Serving Layer (FastAPI REST API)
- **Component:** `src/api/app.py` and `src/api/schemas.py`
- **Responsibility:** Provides high-performance asynchronous REST endpoints (`/health`, `/model-info`, `/predict`, `/predict-batch`, `/api/metrics`, `/api/dataset-stats`) with strict Pydantic runtime schema validation.
- **Why it matters:** Microservice architecture ensures decoupled consumption. Mobile apps, web frontends, or batch ETL scripts can consume predictions via standard JSON HTTP requests.

### 2.7. User Presentation Layer (Glassmorphism Dashboard)
- **Component:** `src/api/static/index.html`, `styles.css`, `app.js`
- **Responsibility:** Delivers a modern, single-page web dashboard with 7 specialized sections, dark/light theme support, interactive prediction forms with presets, and dynamic Chart.js visualizations.
- **Why it matters:** Provides an intuitive interface for university examiners, evaluators, and end users to visualize the machine learning system in action without using CLI tools.

### 2.8. Quality Assurance & CI/CD Layer
- **Component:** `tests/` and `.github/workflows/ci-cd.yml`
- **Responsibility:** 21 automated Pytest tests validating dataset integrity, model behavior, probability boundaries, and API error handling. GitHub Actions executes this test harness automatically on every push or pull request to the GitHub repository.
- **Why it matters:** Enforces regression testing so that future modifications cannot silently break existing functionality.

### 2.9. Containerization Layer (Docker)
- **Component:** `Dockerfile` and `.dockerignore`
- **Responsibility:** Packages the Debian-based Python 3.11 environment, dependencies, source files, and model artifacts into a container image.
- **Why it matters:** Guarantees uniform execution across Windows, Linux, macOS, and cloud hosts, eliminating environment inconsistencies.
