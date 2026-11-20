# ACADEMIC PROJECT REPORT
## Iris Flower Classification Using MLOps
**Degree:** Bachelor of Technology (B.Tech) in Artificial Intelligence and Data Science  
**Academic Year:** Third-Year Engineering (T.Y. AIDS)  
**Author:** Nikhil Landage  
**Institution:** Government College of Engineering, Kolhapur  

---

## 1. ABSTRACT
In contemporary software engineering and applied artificial intelligence, transitioning machine learning models from theoretical prototypes into robust, reproducible, and scalable production environments presents a significant challenge. This project, titled **"Iris Flower Classification Using MLOps"**, develops a complete, production-grade Machine Learning Operations (MLOps) architecture for predicting the botanical species of an Iris flower (*Iris Setosa*, *Iris Versicolor*, and *Iris Virginica*) based on four morphological measurements.

The engineering implementation incorporates Data Version Control (**DVC**) for reproducible pipeline tracking, **MLflow** for experiment and artifact logging, and **Scikit-learn** for training and evaluating both a primary **Decision Tree Classifier** and a comparative **Random Forest Classifier**. A strict stratified splitting protocol ($80\%$ train, $20\%$ test) prevents data leakage. The model selection engine evaluates holdout metrics and serializes the winning model (**Random Forest**, achieving $96.67\%$ test accuracy and $96.66\%$ weighted F1-score) using **joblib**. An asynchronous **FastAPI** backend exposes validated REST endpoints with **Pydantic** data schemas, accompanied by an interactive, responsive glassmorphism dashboard built with **HTML5, CSS3, JavaScript, and Chart.js**. Automated unit and integration testing is executed via **Pytest** ($21$ passing tests) and orchestrated across a **GitHub Actions** CI/CD pipeline, fully packaged within a portable **Docker** container.

---

## 2. INTRODUCTION
Machine Learning Operations (MLOps) represents a paradigm shift that integrates Machine Learning system development (Dev) with operational execution (Ops). Historically, machine learning coursework has emphasized algorithmic modeling inside static Jupyter notebooks. However, real-world deployment requires answering critical operational questions:
1. How do we track iterations of datasets and pipeline stages deterministically?
2. How do we prevent subtle data leakage during preprocessing?
3. How do we log hyperparameters, metrics, and models to ensure complete auditability?
4. How do we automate regression testing and continuously integrate new code?
5. How do we expose model predictions efficiently to end-user interfaces?

This project addresses these challenges by building an end-to-end, reproducible MLOps system around the canonical Iris dataset.

---

## 3. PROBLEM DEFINITION
Given four continuous numerical features describing a specimen's floral dimensions:
1. Sepal Length ($x_1 \in \mathbb{R}^+$)
2. Sepal Width ($x_2 \in \mathbb{R}^+$)
3. Petal Length ($x_3 \in \mathbb{R}^+$)
4. Petal Width ($x_4 \in \mathbb{R}^+$)

The mathematical objective is to learn a mapping function $f: \mathbf{x} \to y$, where $y \in \{0, 1, 2\}$ represents one of the three discrete botanical classes:
- $y = 0$: *Iris-setosa*
- $y = 1$: *Iris-versicolor*
- $y = 2$: *Iris-virginica*

In addition to algorithmic classification, the operational objective is to ensure that the entire data loading, preprocessing, training, tracking, testing, and deployment lifecycle is automated, versioned, containerized, and accessible via standard REST protocols.

---

## 4. PROJECT OBJECTIVES
1. **Pipeline Determinism & Versioning:** Implement DVC to track raw and processed data, configure a multi-stage dependency pipeline (`dvc.yaml`), and ensure bit-for-bit reproducibility.
2. **Data Leakage Prevention:** Strictly decouple the training partition ($n=120$) from the testing partition ($n=30$) prior to model fitting, applying stratified sampling with a constant random seed ($42$).
3. **Comparative Machine Learning:** Train and tune both a single-tree baseline (**Decision Tree**) and an ensemble model (**Random Forest**), computing actual multiclass evaluation metrics.
4. **Experiment Governance:** Track all experimental iterations, parameters, training/testing metrics, and confusion matrix artifacts in a local SQLite-backed MLflow tracking store.
5. **Robust API Serving:** Develop a high-throughput, asynchronous FastAPI backend featuring strict Pydantic input validation, Swagger OpenAPI documentation (`/docs`), and health checks.
6. **Modern Frontend Interface:** Construct a glassmorphism single-page web dashboard offering live inference, dataset statistics, multi-metric comparisons, and interactive Chart.js visualizations.
7. **Continuous Integration & Quality Assurance:** Formulate a comprehensive Pytest suite containing 21 tests running automatically on push and pull requests through GitHub Actions.
8. **Containerization:** Package application dependencies, static assets, and serialized models inside a minimal Docker container.

---

## 5. LITERATURE & BACKGROUND OVERVIEW
The Iris dataset was introduced in 1936 by Sir Ronald A. Fisher as an example of linear discriminant analysis. While *Iris Setosa* is linearly separable from the other two species due to its notably compact petal dimensions, *Iris Versicolor* and *Iris Virginica* exhibit non-linear boundary overlap, providing an ideal benchmark for evaluating non-linear tree-based algorithms.

In standard literature:
- **Decision Trees (Breiman et al., 1984):** Partition feature space recursively via orthogonal splits based on impurity criteria such as Gini Impurity:
  $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
  While decision trees provide human-interpretable decision boundaries, they are susceptible to high variance and overfitting when tree depth increases.
- **Random Forests (Breiman, 2001):** Mitigate the variance of individual decision trees by constructing an ensemble of decorrelated trees trained on bootstrap samples (bagging) with random feature subspace selection. The final classification prediction is determined via majority vote or average class probabilities:
  $$\hat{P}(y = c \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(y = c \mid \mathbf{x})$$

Modern MLOps frameworks like DVC and MLflow extend these foundational algorithms into production software systems by enforcing software engineering best practices.

---

## 6. DATASET DESCRIPTION
The dataset utilized is Scikit-learn's built-in Iris dataset:
- **Sample Count:** 150 instances
- **Attribute Count:** 4 numerical attributes (measured in centimeters)
- **Target Count:** 3 balanced classes (50 instances per class)
- **Missing Values:** Zero missing, null, or erroneous entries
- **Feature Statistics:**
  - Sepal Length: Mean $5.84\text{ cm}$, Std $0.83\text{ cm}$, Min $4.3\text{ cm}$, Max $7.9\text{ cm}$
  - Sepal Width: Mean $3.06\text{ cm}$, Std $0.44\text{ cm}$, Min $2.0\text{ cm}$, Max $4.4\text{ cm}$
  - Petal Length: Mean $3.76\text{ cm}$, Std $1.77\text{ cm}$, Min $1.0\text{ cm}$, Max $6.9\text{ cm}$
  - Petal Width: Mean $1.20\text{ cm}$, Std $0.76\text{ cm}$, Min $0.1\text{ cm}$, Max $2.5\text{ cm}$

---

## 7. METHODOLOGY & PIPELINE STAGES

### Stage 1: Data Ingestion (`src/data/load_data.py`)
Loads raw arrays from Scikit-learn, converts them into a standardized Pandas DataFrame with consistent snake_case column names (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `target`, `species_name`), and saves the result to `data/raw/iris.csv`.

### Stage 2: Data Validation (`src/data/validate_data.py`)
Applies defensive data quality gates verifying:
- Exactly 150 rows and 5 core columns.
- Zero NaN or null entries across all rows.
- Data types are strictly positive floating-point numbers.
- Target column values are $\{0, 1, 2\}$ with exactly 50 samples each.

### Stage 3: Exploratory Data Analysis (`src/eda/generate_eda.py`)
Generates 8 high-resolution charts saved to `reports/figures/` and `src/api/static/figures/`:
1. Class distribution bar chart.
2. Sepal length distribution with KDE.
3. Sepal width distribution with KDE.
4. Petal length distribution with KDE.
5. Petal width distribution with KDE.
6. Feature correlation heatmap (Pearson correlation).
7. Petal length vs Petal width scatter plot.
8. Pairwise multivariate feature comparison grid.

### Stage 4: Preprocessing & Partitioning (`src/features/preprocess.py`)
Applies stratified sampling to divide the dataset into:
- **Training Set (`data/processed/train.csv`):** 120 samples (40 Setosa, 40 Versicolor, 40 Virginica).
- **Testing Set (`data/processed/test.csv`):** 30 samples (10 Setosa, 10 Versicolor, 10 Virginica).
- Random seed is fixed to 42 to guarantee deterministic, leak-free reproducibility.

### Stage 5: Model Training, Evaluation, & MLflow Tracking (`src/models/train.py`)
Trains both models with configured hyperparameters:
- Decision Tree: `criterion='gini'`, `max_depth=4`, `min_samples_split=2`, `min_samples_leaf=1`.
- Random Forest: `n_estimators=100`, `criterion='gini'`, `max_depth=4`, `min_samples_split=2`, `min_samples_leaf=1`.
- MLflow logs hyperparameters, train/test metrics, confusion matrix figures, and Scikit-learn model artifacts to `Iris_Classification_Experiment`.
- Automatically selects the model with higher weighted F1-score and serializes `models/best_model.joblib` and `models/model_metadata.json`.

---

## 8. SYSTEM ARCHITECTURE
The system operates as a layered architecture:
1. **Storage Layer:** DVC tracking raw and processed CSVs alongside joblib binary artifacts.
2. **Governance Layer:** MLflow tracking store backed by SQLite (`mlflow.db`).
3. **Application Layer:** FastAPI ASGI server providing asynchronous request routing and Pydantic validation.
4. **Presentation Layer:** Glassmorphism web dashboard serving interactive UI, Chart.js visualizations, and live REST inference.
5. **Quality Layer:** Pytest automated test harness running locally and within GitHub Actions CI/CD.
6. **Container Layer:** Docker container packaging Python 3.11, dependencies, and application assets.

---

## 9. IMPLEMENTATION DETAILS

### DVC Configuration (`dvc.yaml`)
Defined with 5 interdependent stages: `load_data`, `validate_data`, `eda`, `preprocess`, and `train_and_evaluate`. Executing `dvc repro` reads hashes from `dvc.lock` and skips unchanged stages, preventing redundant computation.

### FastAPI Implementation (`src/api/app.py`)
- `GET /`: Serves the single-page dashboard.
- `GET /health`: Checks API health, uptime, and model loading status.
- `GET /model-info`: Returns active model metadata, accuracy, F1-score, and hyperparameters.
- `POST /predict`: Performs single-flower inference, calculating class probabilities.
- `POST /predict-batch`: Accepts a list of feature sets for high-throughput batch classification.
- `GET /api/metrics`: Delivers comparative metrics to populate dashboard charts.
- `GET /api/dataset-stats`: Supplies descriptive dataset statistics and sample rows.

---

## 10. EXPERIMENTAL RESULTS

Empirical evaluation on the holdout test set ($n=30$):

| Evaluation Metric | Decision Tree Classifier | Random Forest Classifier | Delta |
| :--- | :---: | :---: | :---: |
| **Training Accuracy** | $99.17\%$ | $100.0\%$ | $+0.83\%$ |
| **Test Accuracy** | $93.33\%$ | **$96.67\%$** | **$+3.34\%$** |
| **Test Precision (Weighted)** | $93.33\%$ | **$96.97\%$** | **$+3.64\%$** |
| **Test Precision (Macro)** | $93.33\%$ | **$96.97\%$** | **$+3.64\%$** |
| **Test Recall (Weighted)** | $93.33\%$ | **$96.67\%$** | **$+3.34\%$** |
| **Test Recall (Macro)** | $93.33\%$ | **$96.67\%$** | **$+3.34\%$** |
| **Test F1-Score (Weighted)** | $93.33\%$ | **$96.66\%$** | **$+3.33\%$** |
| **Test F1-Score (Macro)** | $93.33\%$ | **$96.66\%$** | **$+3.33\%$** |

### Confusion Matrix Breakdown (Test Partition, $n=30$):
- **Decision Tree:**
  - Setosa: 10/10 correct
  - Versicolor: 9/10 correct (1 misclassified as Virginica)
  - Virginica: 9/10 correct (1 misclassified as Versicolor)
  - *Total Errors:* 2 misclassifications.
- **Random Forest:**
  - Setosa: 10/10 correct
  - Versicolor: 9/10 correct (1 misclassified as Virginica)
  - Virginica: 10/10 correct (0 misclassifications)
  - *Total Errors:* 1 misclassification.

**Decision:** The Random Forest Classifier was automatically designated as the production model due to superior generalization and lower variance.

---

## 11. TESTING & QUALITY ASSURANCE
A comprehensive automated test suite consisting of 21 tests was implemented using **Pytest**:
- **Dataset Tests (6):** Verified file existence, shape dimensions $(150, 5)$, non-null constraints, numerical ranges, class balance, and schema validation.
- **Machine Learning Tests (5):** Verified artifact generation, deserialization via joblib, valid class prediction indexes $\{0, 1, 2\}$, probability axiom compliance ($\sum p_i = 1.0$), and Setosa prediction correctness.
- **API Tests (10):** Verified root, health, model-info, single prediction, batch prediction, metrics, dataset stats, and HTTP 422 validation rejection for negative values, missing fields, and invalid string types.

All 21 tests execute cleanly in under 5 seconds with zero failures.

---

## 12. ADVANTAGES
1. **Auditability & Traceability:** Every experiment, parameter change, and metric is recorded in MLflow.
2. **Reproducibility:** DVC ensures any engineer can clone the repository and reproduce the exact model with a single `dvc repro` command.
3. **Production-Ready Architecture:** Clean separation of concerns across ingestion, preprocessing, modeling, API serving, and UI.
4. **Input Protection:** Pydantic schemas enforce type safety and reject out-of-boundary physical measurements before inference execution.
5. **No Cloud Dependency Required:** The entire stack operates 100% locally on Windows 11 with zero paid cloud subscriptions.

---

## 13. LIMITATIONS
1. **Dataset Scope:** The Iris dataset contains 150 instances, which, while ideal for benchmarking and academic demonstrations, does not exhibit large-scale big data challenges.
2. **Local Storage:** The current DVC cache is stored on the local filesystem rather than a distributed cloud bucket (e.g., S3).
3. **Static Model:** Model retraining requires triggering the pipeline rather than streaming continuous feedback from production inference logs.

---

## 14. FUTURE SCOPE
1. **Data Drift & Concept Drift Monitoring:** Implement Evidently AI or Great Expectations to trigger alerts when input distributions shift.
2. **Distributed Cloud DVC Remotes:** Connect DVC to an enterprise Amazon S3 or Google Cloud Storage bucket.
3. **Continuous Retraining (CT) Pipeline:** Establish automated retraining jobs scheduled via Apache Airflow or GitHub Actions cron jobs.
4. **Kubernetes Orchestration:** Deploy Docker containers across Kubernetes clusters with automated horizontal pod autoscaling and Prometheus/Grafana monitoring.

---

## 15. CONCLUSION
The **Iris Flower Classification Using MLOps** capstone project successfully demonstrates how foundational machine learning can be elevated to production standards through modern DevOps and MLOps practices. By combining DVC, MLflow, Scikit-learn, FastAPI, Pytest, Docker, and GitHub Actions, this project delivers a technically sound, reproducible, and professionally presented machine learning system suited for university evaluation and industrial application.
