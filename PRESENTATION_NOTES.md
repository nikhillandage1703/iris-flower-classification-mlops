# 🎤 PROJECT PRESENTATION & LIVE DEMO SCRIPT
## Iris Flower Classification Using MLOps
**Degree:** B.Tech Artificial Intelligence and Data Science (T.Y. AIDS)  
**Presenter:** Yash Mimrot  
**Institution:** Government College of Engineering, Kolhapur  

---

## 1. INTRODUCTION SPEECH (Approx. 2 Minutes)
> *"Respected external examiner, internal guide, and fellow peers, good morning.*
>
> *Today, I am proud to present my Third-Year Engineering capstone project titled **'Iris Flower Classification Using MLOps'**.*
>
> *In traditional machine learning academics, projects often end inside an exploratory Jupyter notebook once a high accuracy percentage is displayed. However, in industry, model training accounts for only a small percentage of a real-world machine learning system. The real engineering begins with **reproducibility, data versioning, experiment tracking, automated regression testing, API integration, and containerized deployment**.*
>
> *In this project, I have built an end-to-end MLOps pipeline using the Iris dataset. Using **DVC** for data version control, **MLflow** for experiment tracking, **Scikit-learn** for modeling, **FastAPI** for asynchronous backend serving, **Pytest** with **GitHub Actions** for CI/CD, and **Docker** for containerization, this project represents a complete, auditable software system designed from the ground up."*

---

## 2. MODULE-BY-MODULE EXPLANATION (Approx. 3 Minutes)

### Module 1: Data Ingestion, Quality Gate & DVC
- **What to say:**
  > *"First, in `src/data/load_data.py`, we ingest the canonical 150-sample Iris dataset from Scikit-learn and standardize all column names. Immediately following ingestion, `src/data/validate_data.py` acts as a quality gate, testing for zero missing values, positive floating-point measurements, and equal class balance.*
  >
  > *All stages are version-controlled using **DVC** in `dvc.yaml`. When we run `dvc repro`, DVC checks hash fingerprints and only runs stages whose code or data have changed."*

### Module 2: Preprocessing & Leakage Prevention
- **What to say:**
  > *"To ensure academic and technical rigor, we enforce **zero data leakage**. In `src/features/preprocess.py`, the dataset is split using stratified sampling with a fixed random seed of 42. 120 samples are allocated to training, and 30 samples are strictly isolated for the holdout test partition."*

### Module 3: Comparative ML & MLflow Tracking
- **What to say:**
  > *"We implemented two machine learning algorithms: a primary **Decision Tree Classifier** and an ensemble **Random Forest Classifier**. Both are governed through **MLflow** under the experiment `Iris_Classification_Experiment`.*
  >
  > *Our training pipeline logs all hyperparameters, train/test metrics, confusion matrix figures, and serialized model files to a local SQLite tracking database. Our automated selector compares the weighted F1-scores on the unseen test set, selects the best model (**Random Forest** at $96.67\%$ accuracy), and serializes it using `joblib`."*

### Module 4: FastAPI REST Backend & Glassmorphism Dashboard
- **What to say:**
  > *"To serve predictions in real-time, we built an asynchronous **FastAPI** backend in `src/api/app.py`. It provides input validation using Pydantic, swagger documentation at `/docs`, and directly serves our custom-designed glassmorphism web dashboard featuring real-time Chart.js telemetry."*

### Module 5: Automated Testing & CI/CD
- **What to say:**
  > *"We wrote 21 automated unit and integration tests using **Pytest**. Our GitHub Actions workflow (`ci-cd.yml`) guarantees that any push or pull request automatically reproduces the pipeline and runs all 21 tests on an Ubuntu runner, preventing any broken code from being merged."*

---

## 3. LIVE DEMO SEQUENCE (Step-by-Step for Viva)

Follow this exact sequence on your laptop during the viva demonstration:

### Step 1: Show the Terminal & Project Cleanliness
Open PowerShell and show that the virtual environment is active:
```powershell
Get-Location
```

### Step 2: Demonstrate DVC Reproducibility
Run DVC status to show everything is version-controlled:
```powershell
dvc status
```
> *Say to examiner: "As you can see, DVC confirms that our data, parameters, and pipeline stages are completely synchronized and up to date."*

### Step 3: Run the 21 Automated Pytest Checks
Execute the test suite in verbose mode:
```powershell
pytest -v tests/
```
> *Point to screen: "All 21 tests pass in under 5 seconds, validating dataset schema, model predictions, probability sums, and API error handling."*

### Step 4: Show MLflow Experiment Tracking
Launch MLflow:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
- Open `http://localhost:5000` in Chrome/Edge.
- Show the `Iris_Classification_Experiment`.
- Show `Decision_Tree_Run` vs `Random_Forest_Run`.
- Click into artifacts and display the saved confusion matrices.

### Step 5: Launch FastAPI & The Interactive Dashboard
In a PowerShell window, start Uvicorn:
```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000`:
1. **Overview Tab:** Show real-time telemetry, 150 samples, 4 features, and production accuracy of $96.7\%$.
2. **Dataset Explorer:** Show the live preview table and generated EDA distribution plots.
3. **Model Comparison:** Show the comparison table and multi-metric Chart.js graph.
4. **Model Evaluation:** Show the confusion matrix comparisons and explain why Setosa has 10/10 accuracy.
5. **Flower Prediction Lab (Live Test):**
   - Click **Setosa Sample** preset $\to$ Click **Predict Species** $\to$ Show $98.5\%$ confidence and blue badge.
   - Click **Versicolor Sample** preset $\to$ Click **Predict Species** $\to$ Show green badge.
   - Click **Virginica Sample** preset $\to$ Click **Predict Species** $\to$ Show amber badge.
6. **Swagger UI:** Click the top-right button **Swagger Docs** to open `http://localhost:8000/docs` and demonstrate the interactive OpenAPI documentation.

---

## 4. RESULTS EXPLANATION (Key Talking Points)
- **Decision Tree Result:** $93.33\%$ test accuracy, $93.33\%$ F1-score (2 test errors).
- **Random Forest Result:** $96.67\%$ test accuracy, $96.66\%$ F1-score (only 1 test error).
- **Why Random Forest won:** By training 100 decorrelated trees on bootstrap subsets and averaging their predictions, Random Forest smoothed out decision boundaries and eliminated the false prediction on Virginica that the single Decision Tree suffered from.

---

## 5. CONCLUSION SPEECH (Approx. 1 Minute)
> *"In conclusion, this project demonstrates that modern machine learning is not merely about finding a mathematical function, but about establishing an automated, auditable, and reliable engineering lifecycle.*
>
> *Through DVC, MLflow, FastAPI, Pytest, Docker, and GitHub Actions, we have demonstrated a comprehensive, production-ready system. Thank you for your time, and I am now ready to answer any questions."*
