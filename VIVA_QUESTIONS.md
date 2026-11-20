# 🎓 VIVA VOCE & ORAL EXAMINATION PREPARATION GUIDE
## Iris Flower Classification Using MLOps
**Project Level:** Third-Year Engineering (T.Y. AIDS)  
**Author:** Nikhil Landage  
**Institution:** Government College of Engineering, Kolhapur  

---

### Category 1: Problem Domain & Dataset

#### Q1. What is the objective of this project?
**Answer:** The objective is to build a complete, production-grade MLOps system that predicts the botanical species of an Iris flower (*Iris Setosa*, *Iris Versicolor*, or *Iris Virginica*) based on four continuous numerical morphological measurements, while demonstrating data versioning, experiment tracking, automated testing, API serving, and containerization.

#### Q2. What are the four input features and three target classes of the Iris dataset?
**Answer:**
- **Input Features (all in cm):** `sepal_length`, `sepal_width`, `petal_length`, and `petal_width`.
- **Target Classes:** `0`: *Iris Setosa*, `1`: *Iris Versicolor*, `2`: *Iris Virginica*.

#### Q3. How many samples exist in the dataset, and what is its class distribution?
**Answer:** The dataset contains 150 samples in total, perfectly balanced with 50 samples for each of the three species.

#### Q4. Which Iris species is linearly separable, and which two overlap?
**Answer:** *Iris Setosa* is linearly separable from the other two species due to its significantly smaller petal length and width. *Iris Versicolor* and *Iris Virginica* exhibit partial non-linear overlap in feature space.

#### Q5. What is data leakage, and how did you prevent it in this project?
**Answer:** Data leakage occurs when information from the test dataset unintentionally influences model training (e.g., fitting scalers on the entire dataset before splitting). We prevented it by performing a strict stratified split into 120 training and 30 testing samples before fitting any models, using a fixed random seed (`42`).

---

### Category 2: Machine Learning & Algorithms

#### Q6. What algorithms did you implement, and which one is the primary model?
**Answer:** We implemented a **Decision Tree Classifier** as the primary interpretable baseline model and a **Random Forest Classifier** as the comparison ensemble model.

#### Q7. What is Gini Impurity, and how is it calculated?
**Answer:** Gini impurity measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the class distribution.
Formula:
$$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
A Gini impurity of $0$ indicates a pure node where all samples belong to a single class.

#### Q8. Why did Random Forest outperform Decision Tree?
**Answer:** Standalone Decision Trees tend to overfit and exhibit high variance. Random Forest is an ensemble of 100 decorrelated decision trees trained on bootstrap samples with random feature subsets. By averaging tree predictions, it reduces variance and achieves superior test generalization ($96.67\%$ accuracy vs $93.33\%$).

#### Q9. What is the difference between Weighted Averaging and Macro Averaging for multiclass metrics?
**Answer:**
- **Macro Averaging:** Computes the metric independently for each class and calculates their unweighted mean. Every class is treated equally.
- **Weighted Averaging:** Computes the metric for each class and calculates their average weighted by the number of true instances (support) of each class.
Because our test split is balanced (10 samples per class), macro and weighted averages produce identical results.

#### Q10. What is a Confusion Matrix?
**Answer:** A confusion matrix is an $N \times N$ table (where $N=3$ for Iris) that visualizes model classification performance by showing counts of True Positives, False Positives, True Negatives, and False Negatives for each species.

#### Q11. What were the exact test results of your models?
**Answer:** On the 30-sample holdout test partition:
- **Decision Tree:** $93.33\%$ test accuracy, $93.33\%$ weighted F1-score (2 errors: 1 Versicolor misclassified as Virginica, 1 Virginica misclassified as Versicolor).
- **Random Forest:** $96.67\%$ test accuracy, $96.66\%$ weighted F1-score (only 1 error: 1 Versicolor misclassified as Virginica).

#### Q12. Why did you use `joblib` instead of standard `pickle` for model serialization?
**Answer:** `joblib` is optimized specifically for Python objects containing large NumPy arrays and Scikit-learn estimators, serializing and deserializing scientific pipelines faster and more efficiently than standard `pickle`.

---

### Category 3: Data Version Control (DVC)

#### Q13. What is DVC, and why is it needed in machine learning?
**Answer:** DVC (Data Version Control) is an open-source tool for managing datasets, models, and ML pipelines alongside Git. Git cannot efficiently track large data files or binary model weights; DVC tracks file content hashes in lightweight `.dvc` and `dvc.lock` files, keeping large data outside Git history.

#### Q14. What are `dvc.yaml` and `params.yaml`?
**Answer:**
- **`params.yaml`:** A central configuration file defining tunable parameters (e.g., test split ratio, random seed, tree hyperparameters).
- **`dvc.yaml`:** A declarative pipeline file defining stages (`load_data`, `validate_data`, `eda`, `preprocess`, `train_and_evaluate`), their command execution, input dependencies (`deps`), parameters (`params`), and generated outputs (`outs`).

#### Q15. What does the command `dvc repro` do?
**Answer:** `dvc repro` executes the pipeline stages defined in `dvc.yaml`. It checks whether dependencies or parameters have changed since the last execution. If nothing changed, it skips those stages, saving computational time.

#### Q16. What is the difference between Local DVC and Remote DVC?
**Answer:** Local DVC caches dataset versions and models in a hidden `.dvc/cache` directory on your local disk. Remote DVC pushes this cache to cloud storage (like AWS S3 or Google Cloud Storage) so team members can share data using `dvc pull` and `dvc push`.

---

### Category 4: Experiment Tracking with MLflow

#### Q17. What is MLflow, and what problem does it solve?
**Answer:** MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It solves the chaos of unorganized experiments by automatically tracking hyperparameters, training/testing metrics, artifacts (plots, figures), and code versions in a centralized dashboard.

#### Q18. What is the difference between an MLflow Experiment and an MLflow Run?
**Answer:**
- **Experiment:** A logical grouping of related machine learning runs (e.g., `Iris_Classification_Experiment`).
- **Run:** A single execution of model training within an experiment (e.g., `Decision_Tree_Run` or `Random_Forest_Run`) that logs specific parameters, metrics, and output files.

#### Q19. What backend tracking store did you use for MLflow, and why?
**Answer:** We used SQLite (`sqlite:///mlflow.db`). In MLflow 3+, SQLite is the recommended database backend because it supports concurrent runs, ACID transactions, robust querying, and artifact tracking.

#### Q20. How do you launch the MLflow UI?
**Answer:** By running the command:
`mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
and opening `http://localhost:5000` in the browser.

---

### Category 5: Backend & FastAPI

#### Q21. Why did you choose FastAPI instead of Flask or Django?
**Answer:** FastAPI offers:
1. Native asynchronous (async/await) performance using Starlette and Uvicorn.
2. Automatic data validation using Python type hints and Pydantic.
3. Automatic interactive OpenAPI/Swagger documentation generated at `/docs`.
4. High developer productivity with low boilerplate.

#### Q22. What role does Pydantic play in your API?
**Answer:** Pydantic enforces data type validation and boundary checks at runtime. In `IrisFeatures`, we validate that measurements are strictly positive floats (`gt=0.0`) and within realistic physical boundaries (`lt=20.0`). If an invalid payload is sent, FastAPI automatically rejects it with HTTP 422 Unprocessable Entity.

#### Q23. What are the key endpoints in your API?
**Answer:**
- `GET /`: Serves the interactive dashboard.
- `GET /health`: Health check and model readiness indicator.
- `GET /model-info`: Production model metadata, accuracy, and hyperparameters.
- `POST /predict`: Single flower species prediction with probability breakdown.
- `POST /predict-batch`: High-throughput batch classification for multiple flowers.
- `GET /api/metrics`: Comparative evaluation metrics.
- `GET /api/dataset-stats`: Descriptive dataset statistics and preview rows.

#### Q24. How does the model return class probabilities?
**Answer:** Through Scikit-learn's `.predict_proba()` method. It outputs the fraction of trees in the Random Forest voting for each class, which our API converts into a percentage distribution (e.g., Setosa: $98.5\%$, Versicolor: $1.5\%$, Virginica: $0.0\%$).

---

### Category 6: Automated Testing & CI/CD

#### Q25. What is Pytest, and how many tests did you implement?
**Answer:** Pytest is a Python testing framework for writing simple, scalable unit and integration tests. We implemented 21 automated tests across three modules: `test_dataset.py` (6 tests), `test_model.py` (5 tests), and `test_api.py` (10 tests).

#### Q26. What is a Pytest fixture?
**Answer:** A fixture is a reusable function decorated with `@pytest.fixture` that prepares a baseline environment or mock resource (like loading the dataset or instantiating a FastAPI `TestClient`) before tests execute, and cleans up afterwards.

#### Q27. What is GitHub Actions, and what does your CI/CD workflow do?
**Answer:** GitHub Actions is an automated continuous integration and continuous delivery platform. Our workflow file (`.github/workflows/ci-cd.yml`) triggers on every `push` and `pull_request`, sets up Python 3.11, installs dependencies, reproduces pipeline stages, and runs all 21 Pytest checks to ensure no breaking changes enter the codebase.

---

### Category 7: Docker & Deployment

#### Q28. What is Docker, and why is it used in this project?
**Answer:** Docker packages an application and all its system dependencies, runtime environments, libraries, and model weights into an isolated, self-contained container. This eliminates the "it works on my machine" problem, guaranteeing identical execution on Windows, Linux, macOS, or cloud servers.

#### Q29. What is the purpose of `.dockerignore`?
**Answer:** `.dockerignore` specifies files and directories (such as `.venv/`, `.git/`, `.dvc/cache/`, `__pycache__/`) that should not be copied into the container image, reducing build time and minimizing image size.

#### Q30. What commands build and run your Docker container?
**Answer:**
- Build: `docker build -t iris-mlops-app:latest .`
- Run: `docker run -d -p 8000:8000 --name iris-mlops-container iris-mlops-app:latest`
- View: `docker ps`
- Stop: `docker stop iris-mlops-container`
