# 🛠️ WINDOWS 11 STEP-BY-STEP SETUP GUIDE
## Iris Flower Classification Using MLOps
**Operating System:** Windows 11 (PowerShell)  
**Author:** Yash Mimrot  
**Institution:** Government College of Engineering, Kolhapur  

This guide provides exact, copy-paste PowerShell instructions to configure, run, and test the entire project locally on a Windows 11 machine without requiring paid cloud subscriptions.

---

## 1. Prerequisites Check

Open **Windows Terminal (PowerShell)** and verify installed tools:

```powershell
python --version
git --version
```
- **Recommended Python Version:** Python 3.11, 3.12, or 3.13 (Python 3.13 64-bit is fully supported).
- **Git:** Git for Windows (2.25+ recommended).

---

## 2. Setting PowerShell Execution Policy

By default, Windows 11 restricts running unsigned PowerShell activation scripts. To temporarily allow virtual environment activation for your current PowerShell window without altering system-wide security policies, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 3. Virtual Environment Setup

Navigate to the project root directory:

```powershell
cd d:\iris_mlops_friend
```

Create a virtual environment (if not already created):
```powershell
python -m venv .venv
```

Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(When activated, your prompt will display `(.venv)` in front of the command prompt)*

---

## 4. Installing Project Dependencies

Ensure `pip` is up to date, then install all project packages from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running the Complete MLOps Pipeline with DVC

To reproduce the complete pipeline (data ingestion, validation, exploratory data analysis, stratified preprocessing, and model training):

```powershell
dvc repro
```

Verify that all stages are up to date:
```powershell
dvc status
```
*Expected Output:* `Data and pipelines are up to date.`

---

## 6. Launching the MLflow Experiment Tracking Dashboard

To inspect runs, parameters, metrics, and confusion matrix artifacts:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open your web browser and navigate to:
**`http://localhost:5000`**

---

## 7. Running the Automated Pytest Test Suite

Execute all 21 automated unit and integration tests:

```powershell
pytest -v tests/
```

*Expected Output:* `21 passed in ~4.5s`

---

## 8. Starting the FastAPI Backend & Web Dashboard

Launch the FastAPI application with Uvicorn:

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Access the interfaces in your browser:
- **Interactive Web Dashboard:** `http://localhost:8000`
- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check Endpoint:** `http://localhost:8000/health`

---

## 9. Testing API Predictions via Command Line

Open a second PowerShell window and test the prediction endpoint with `curl.exe`:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'
```

---

## 10. Docker Build and Execution Commands

If Docker Desktop is installed on your Windows machine:

### 1. Build the Docker Image:
```powershell
docker build -t iris-mlops-app:latest .
```

### 2. Run the Container:
```powershell
docker run -d -p 8000:8000 --name iris-mlops-container iris-mlops-app:latest
```

### 3. Check Running Containers:
```powershell
docker ps
```

### 4. View Container Logs:
```powershell
docker logs iris-mlops-container
```

### 5. Stop the Container:
```powershell
docker stop iris-mlops-container
```

---

## 11. Connecting to GitHub (When Ready)

To push this repository to your personal GitHub account:

```powershell
# 1. Review Git status
git status

# 2. Stage all files
git add .

# 3. Commit changes
git commit -m "Complete Iris MLOps engineering system: DVC, MLflow, FastAPI, Pytest, Docker"

# 4. Add your remote GitHub repository URL
git remote add origin https://github.com/<your-username>/iris-flower-classification-mlops.git

# 5. Push to GitHub
git push -u origin main
```
