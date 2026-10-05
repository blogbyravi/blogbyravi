# Machine Learning Model Deployment: From Notebook to AWS EC2

## Abstract

This guide presents a practical path for taking a machine learning model from notebook experimentation to a deployable service on AWS EC2. It covers model persistence, reusable Python code, a Streamlit interface, a FastAPI inference service, version control, cloud setup, and foundational production practices. Each stage builds on the previous one, separating model, application, and infrastructure responsibilities so that the resulting project is easier to test, maintain, and extend. The guide is intended for learners who understand basic Python and machine learning and want a structured introduction to deployment; it uses a small scikit-learn example to make the workflow concrete.

## Table of Contents

- [Abstract](#abstract)

1. [ML Deployment Overview](#1-ml-deployment-overview)
2. [Saving and Loading Trained ML Models](#2-saving-and-loading-trained-ml-models)
3. [From Notebooks to Python Scripts](#3-from-notebooks-to-python-scripts)
4. [Training and Prediction Scripts](#4-training-and-prediction-scripts)
5. [Streamlit App for ML Model](#5-streamlit-app-for-ml-model)
6. [API Development with FastAPI](#6-api-development-with-fastapi)
7. [Deploying the ML Model as an API](#7-deploying-the-ml-model-as-an-api)
8. [FastAPI + Streamlit Integration](#8-fastapi--streamlit-integration)
9. [Git and GitHub](#9-git-and-github)
10. [AWS EC2 - ML API Deployment](#10-aws-ec2---ml-api-deployment)
11. [Production Checklist](#11-production-checklist)
12. [Common Deployment Issues](#12-common-deployment-issues)
13. [End-to-End Workflow](#13-end-to-end-workflow)

---

## 1. ML Deployment Overview

This section defines deployment in the context of the machine learning lifecycle. It introduces common hosting options and establishes the progression from an experimental notebook to an application that can serve predictions.

### 1.1 What is ML model deployment?

Machine learning deployment is the process of making a trained ML model available for real-world predictions.

A typical ML workflow is:

```text
Business Problem
       |
       v
Data Collection
       |
       v
Data Cleaning & EDA
       |
       v
Feature Engineering
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
Save Model
       |
       v
Application / API
       |
       v
Deployment
       |
       v
Monitoring & Maintenance
```

Training a model in Jupyter Notebook is only one part of the ML lifecycle.

For example, suppose we train a model to predict house prices.

The notebook may contain:

```text
Input data
   ↓
Data preprocessing
   ↓
Feature engineering
   ↓
Model training
   ↓
Evaluation
```

Deployment adds another layer:

```text
User / Application
       ↓
API or Web App
       ↓
Preprocessing
       ↓
Trained ML Model
       ↓
Prediction
       ↓
Response
```

### 1.2 Why do we need deployment?

A trained model has limited business value if nobody can use it.

Deployment allows:

- Applications to call the model.
- Users to submit input data.
- Other systems to consume predictions.
- Business teams to use ML results.
- Models to be integrated into production workflows.

### 1.3 Common deployment options

| Approach | Use case |
|---|---|
| Streamlit | Quick ML demos and internal applications |
| FastAPI | Production-style prediction APIs |
| Flask | Lightweight APIs |
| Docker | Consistent packaging and deployment |
| AWS EC2 | Deploy applications on a virtual server |
| AWS SageMaker | Managed ML training/deployment |
| Kubernetes | Large-scale container orchestration |

For learning and portfolio projects, a useful progression is:

```text
Jupyter Notebook
      ↓
Python Scripts
      ↓
Saved Model
      ↓
Streamlit
      ↓
FastAPI
      ↓
FastAPI + Streamlit
      ↓
GitHub
      ↓
AWS EC2
```

---

## 2. Saving and Loading Trained ML Models

This section explains how to persist a trained estimator and restore it for inference. It also emphasizes saving preprocessing with the estimator, versioning model artifacts, and recording the information needed to reproduce a deployment.

Once the model is trained, we normally save it to a file.

Two common approaches are:

- `joblib`
- `pickle`

For scikit-learn models, `joblib` is commonly used.

### 2.1 Install joblib

```bash
pip install joblib
```

### 2.2 Train a model

Example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
```

### 2.3 Save the model

```python
import joblib

joblib.dump(model, "model.pkl")
```

The file:

```text
model.pkl
```

contains the trained model.

### 2.4 Load the model

```python
import joblib

model = joblib.load("model.pkl")
```

Then:

```python
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

print(prediction)
```

### 2.5 Important: Save preprocessing too

One of the most common deployment mistakes is saving only the model while forgetting preprocessing.

Suppose training uses:

```text
Raw Data
   ↓
StandardScaler
   ↓
RandomForest
```

During prediction, we must perform the same transformation:

```text
New Data
   ↓
Same StandardScaler
   ↓
RandomForest
   ↓
Prediction
```

A better approach is to use a scikit-learn `Pipeline`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model_pipeline.pkl")
```

Now the saved object contains both preprocessing and the model.

Load it:

```python
pipeline = joblib.load("model_pipeline.pkl")

prediction = pipeline.predict(new_data)
```

This reduces the chance of training-serving inconsistencies.

### 2.6 Model artifact best practices

Do not blindly overwrite production models.

Consider versioning:

```text
models/
├── model_v1.pkl
├── model_v2.pkl
└── model_v3.pkl
```

Also record:

- Python version
- Library versions
- Training dataset version
- Features used
- Model algorithm
- Evaluation metrics
- Training date
- Model version

Example:

```text
model_version: 1.0
algorithm: RandomForestClassifier
accuracy: 0.95
python: 3.12
scikit-learn: 1.x
```

> Security note: `pickle` and `joblib` model files should only be loaded from trusted sources. Do not load arbitrary model files from untrusted locations.

---

## 3. From Notebooks to Python Scripts

This section shows why exploratory notebooks should be complemented by reusable application code. It introduces a project layout and separates training, prediction, API, and interface responsibilities.

### 3.1 Why move from notebooks?

Jupyter Notebooks are excellent for:

- Exploration
- Visualization
- Experimentation
- EDA
- Model experimentation

However, production applications benefit from Python modules and scripts because they provide:

- Reusability
- Version control
- Testing
- Maintainability
- Automation
- Easier deployment

Instead of keeping everything in:

```text
model_training.ipynb
```

we can create:

```text
ml-project/
│
├── data/
├── models/
├── notebooks/
├── src/
├── app/
├── requirements.txt
└── README.md
```

### 3.2 Recommended project structure

A simple project can look like:

```text
ml-deployment-project/
│
├── data/
│   └── iris.csv
│
├── notebooks/
│   └── model_experiment.ipynb
│
├── models/
│   └── model_pipeline.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── api/
│   └── main.py
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### 3.3 Separate responsibilities

A useful design is:

```text
train.py
    |
    |-- Load data
    |-- Clean data
    |-- Train model
    |-- Evaluate model
    |-- Save model

predict.py
    |
    |-- Load model
    |-- Validate input
    |-- Predict
    |-- Return result

main.py
    |
    |-- API endpoints
    |-- Request validation
    |-- Call prediction logic

app.py
    |
    |-- User interface
    |-- Call API
    |-- Display prediction
```

This separation makes the application easier to maintain.

---

## 4. Training and Prediction Scripts

This section turns the proposed project structure into working training and inference scripts. The separation makes model creation repeatable and gives the API and user interface a clear prediction function to call.

### 4.1 Training script

Create:

```text
src/train.py
```

Example:

```python
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    data = load_iris()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")

    joblib.dump(
        pipeline,
        "models/model_pipeline.pkl"
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()
```

Run:

```bash
python src/train.py
```

### 4.2 Prediction script

Create:

```text
src/predict.py
```

Example:

```python
import joblib


MODEL_PATH = "models/model_pipeline.pkl"

model = joblib.load(MODEL_PATH)


def predict(features):
    prediction = model.predict([features])

    return int(prediction[0])


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict(sample)

    print("Prediction:", result)
```

Run:

```bash
python src/predict.py
```

### 4.3 Why this separation matters

Without separation:

```text
Notebook
 ├── Training
 ├── Prediction
 ├── Visualization
 ├── API
 └── UI
```

With separation:

```text
Training → Model Artifact
Prediction → Prediction Function
API → Prediction Service
UI → User Interface
```

This is much easier to deploy.

---

## 5. Streamlit App for ML Model

This section builds a lightweight interactive interface for collecting features and displaying model predictions. It is suited to demonstrations and prototypes, and sets up the distinction between a directly embedded model and a separately hosted inference API.

### 5.1 What is Streamlit?

Streamlit is a Python framework for quickly creating data and ML web applications.

It is useful for:

- ML demonstrations
- Portfolio projects
- Internal tools
- Model prototypes
- Data applications

### 5.2 Install Streamlit

```bash
pip install streamlit
```

### 5.3 Create the application

Create:

```text
streamlit_app/app.py
```

Example:

```python
import streamlit as st
import joblib


MODEL_PATH = "models/model_pipeline.pkl"

model = joblib.load(MODEL_PATH)


st.title("Iris Flower Prediction")

st.write(
    "Enter flower measurements to predict the iris class."
)

sepal_length = st.number_input(
    "Sepal Length",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    value=0.2
)


if st.button("Predict"):

    features = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(features)[0]

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted class: {classes[prediction]}"
    )
```

### 5.4 Run Streamlit

From the project root:

```bash
streamlit run streamlit_app/app.py
```

Streamlit starts a local web application.

The workflow becomes:

```text
User
  ↓
Streamlit UI
  ↓
Input Features
  ↓
Saved ML Pipeline
  ↓
Prediction
  ↓
Result displayed
```

### 5.5 Limitations of directly loading the model in Streamlit

This approach is excellent for a demo.

However, if multiple applications need the same model, a separate API service is usually cleaner:

```text
Application A ─┐
               ├──> ML API ──> Model
Application B ─┘
```

This leads to FastAPI.

---

## 6. API Development with FastAPI

This section introduces an HTTP interface for model inference. It defines request validation, health and prediction endpoints, and the local commands needed to run and inspect the service.

### 6.1 What is FastAPI?

FastAPI is a modern Python framework for building APIs.

It is commonly used for ML inference services because it provides:

- Request validation
- Type hints
- Automatic API documentation
- High performance
- Easy integration with Python ML libraries

### 6.2 Install FastAPI and Uvicorn

```bash
pip install fastapi uvicorn
```

### 6.3 Create FastAPI application

Create:

```text
api/main.py
```

Example:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="Iris ML Prediction API",
    version="1.0.0"
)


model = joblib.load(
    "models/model_pipeline.pkl"
)


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "Iris ML API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: IrisRequest):

    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]

    prediction = model.predict(features)[0]

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    return {
        "prediction": int(prediction),
        "class": classes[prediction]
    }
```

### 6.4 Run the API

From the project root:

```bash
uvicorn api.main:app --reload
```

The application can then be accessed locally.

FastAPI also provides interactive API documentation through:

```text
/docs
```

and an alternative documentation interface through:

```text
/redoc
```

### 6.5 Understanding the API request

Example JSON:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

The API performs:

```text
JSON Request
     ↓
Pydantic Validation
     ↓
Feature Preparation
     ↓
ML Model
     ↓
Prediction
     ↓
JSON Response
```

---

## 7. Deploying the ML Model as an API

This section verifies the API locally before cloud deployment. It covers interactive and command-line testing, dependency management, and environment reproducibility so deployment issues can be caught earlier.

### 7.1 Local deployment first

Before deploying to AWS, verify locally.

Start the API:

```bash
uvicorn api.main:app --reload
```

Test:

```text
GET /
GET /health
POST /predict
```

### 7.2 Test using Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

Select:

```text
POST /predict
```

Click:

```text
Try it out
```

Provide:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

Execute the request.

Expected response:

```json
{
    "prediction": 0,
    "class": "Setosa"
}
```

### 7.3 Test using curl

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

### 7.4 Requirements file

Create:

```text
requirements.txt
```

Example:

```text
fastapi
uvicorn[standard]
scikit-learn
pandas
numpy
joblib
streamlit
requests
```

It is better to pin versions for reproducibility in a real deployment:

```text
fastapi==<tested-version>
uvicorn[standard]==<tested-version>
scikit-learn==<tested-version>
joblib==<tested-version>
```

Use versions compatible with the Python environment in which the model was trained.

Generate a requirements file from a controlled virtual environment with:

```bash
pip freeze > requirements.txt
```

Avoid blindly using an environment containing unrelated packages.

---

## 8. FastAPI + Streamlit Integration

This section connects the user interface to the API over HTTP rather than loading the model in the frontend. The resulting separation allows the API to serve multiple clients and lets each component be tested and changed independently.

A stronger architecture separates the UI from the ML API.

### 8.1 Architecture

```text
                 Internet / User
                       |
                       v
              +------------------+
              |    Streamlit     |
              |   Frontend/UI    |
              +------------------+
                       |
                       | HTTP POST
                       v
              +------------------+
              |     FastAPI      |
              |    ML API        |
              +------------------+
                       |
                       v
              +------------------+
              |  ML Model /      |
              |  Pipeline        |
              +------------------+
                       |
                       v
                  Prediction
```

This approach has several advantages:

- UI and API are separated.
- API can be reused by other applications.
- Model logic stays in the backend.
- Frontend can change independently.
- Testing becomes easier.

### 8.2 Streamlit calls FastAPI

Install requests:

```bash
pip install requests
```

Update Streamlit:

```python
import requests
import streamlit as st


st.title("Iris Prediction Application")


sepal_length = st.number_input(
    "Sepal Length",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    value=0.2
)


if st.button("Predict"):

    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload,
        timeout=10
    )

    if response.status_code == 200:

        result = response.json()

        st.success(
            f"Prediction: {result['class']}"
        )

    else:

        st.error(
            f"API error: {response.status_code}"
        )
```

### 8.3 Run both applications

Terminal 1:

```bash
uvicorn api.main:app --reload
```

Terminal 2:

```bash
streamlit run streamlit_app/app.py
```

The complete local architecture is:

```text
Browser
   |
   v
Streamlit
   |
   | HTTP
   v
FastAPI
   |
   v
ML Pipeline
   |
   v
Prediction
```

### 8.4 CORS

If Streamlit and FastAPI are hosted on different origins, configure CORS appropriately.

Example:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

For production, avoid using:

```python
allow_origins=["*"]
```

unless it is genuinely appropriate for the application.

---

## 9. Git and GitHub

This section applies version control to the deployment project. It covers repository setup, a basic commit-and-push workflow, and practical guidance for excluding sensitive credentials and managing model artifacts.

### 9.1 Why Git?

Git provides version control for:

- Python code
- Configuration
- Documentation
- Application changes
- Model development history

GitHub provides remote repository hosting and collaboration.

### 9.2 Initialize Git

From the project root:

```bash
git init
```

Check:

```bash
git status
```

### 9.3 Create .gitignore

Example:

```text
# Python
__pycache__/
*.py[cod]

# Virtual environment
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env

# IDE
.vscode/

# Model files
models/*.pkl

# Data
data/raw/
```

Whether model files should be ignored depends on the project.

For a portfolio project, a small model artifact may be stored separately or through Git LFS/object storage. For larger or sensitive artifacts, use dedicated artifact storage rather than normal Git.

### 9.4 Add files

```bash
git add .
```

Commit:

```bash
git commit -m "Initial ML deployment project"
```

### 9.5 Connect GitHub repository

Create a repository on GitHub.

Then:

```bash
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
```

Rename branch:

```bash
git branch -M main
```

Push:

```bash
git push -u origin main
```

### 9.6 Recommended Git workflow

```text
Make changes
     ↓
git status
     ↓
git add .
     ↓
git commit -m "Meaningful message"
     ↓
git push
```

Example:

```bash
git add .
git commit -m "Add FastAPI prediction endpoint"
git push
```

### 9.7 Never commit secrets

Do not commit:

```text
AWS credentials
API keys
Passwords
Private tokens
.env files
```

Use environment variables or a suitable secret-management solution.

---

## 10. AWS EC2 - ML API Deployment

This section walks through hosting the FastAPI service on a Linux EC2 instance. It covers networking, SSH, dependencies, process management, and a reverse-proxy architecture, while distinguishing a learning deployment from a hardened production service.

### 10.1 What is EC2?

Amazon EC2 provides virtual servers in the AWS cloud.

For an ML API, the basic architecture is:

```text
User
  |
  v
Internet
  |
  v
AWS EC2
  |
  v
FastAPI
  |
  v
ML Model
  |
  v
Prediction
```

This is a good learning deployment because it helps you understand:

- Linux
- Networking
- Security groups
- SSH
- Python environments
- Process management
- API deployment

For production systems, additional components such as a reverse proxy, HTTPS, monitoring, autoscaling, and managed services may be appropriate.

---

### 10.2 Step 1: Create an EC2 instance

In AWS:

```text
AWS Console
   ↓
EC2
   ↓
Launch Instance
```

Choose:

- A suitable Linux AMI.
- An instance size appropriate for your workload.
- A key pair for SSH access.
- A security group.

For a lightweight learning model, a small instance may be enough.

The exact instance type should be selected based on:

- Model size
- RAM requirements
- CPU requirements
- Expected traffic
- Cost

---

### 10.3 Step 2: Configure the security group

At minimum, you typically need:

| Port | Purpose |
|---|---|
| 22 | SSH |
| 8000 | FastAPI testing, if directly exposed |
| 80 | HTTP |
| 443 | HTTPS |

For learning, port `8000` can be exposed temporarily.

For a more production-oriented setup, prefer:

```text
Internet
   ↓
Port 80/443
   ↓
Nginx / Reverse Proxy
   ↓
FastAPI
```

Avoid unnecessarily exposing internal application ports to the entire internet.

Restrict SSH (`22`) to trusted IP addresses where practical.

---

### 10.4 Step 3: Connect to EC2

Using SSH:

```bash
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP
```

The username depends on the AMI.

For Ubuntu, it is commonly:

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

---

### 10.5 Step 4: Update the server

For Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
```

Install Python tools:

```bash
sudo apt install python3 python3-pip python3-venv git -y
```

The exact commands vary by Linux distribution.

---

### 10.6 Step 5: Clone the GitHub repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Enter the project:

```bash
cd ml-deployment-project
```

Check:

```bash
ls
```

---

### 10.7 Step 6: Create virtual environment

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Verify:

```bash
python --version
```

---

### 10.8 Step 7: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```bash
pip list
```

---

### 10.9 Step 8: Verify the model file

Make sure the server can access:

```text
models/model_pipeline.pkl
```

If the model is stored in object storage instead of GitHub, download it during deployment or application startup using an appropriate secure mechanism.

Do not hard-code credentials into the Python source code.

---

### 10.10 Step 9: Run FastAPI

For testing:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The important part is:

```text
--host 0.0.0.0
```

This allows the application to listen on the server's network interface rather than only on localhost.

You can then test:

```text
http://YOUR_EC2_PUBLIC_IP:8000/
```

Swagger:

```text
http://YOUR_EC2_PUBLIC_IP:8000/docs
```

If the security group permits port 8000 and the application is listening correctly, the API should be reachable.

---

### 10.11 Step 10: Keep the API running

Do not rely on a terminal session for a production service.

A simple learning approach can use:

```bash
nohup uvicorn api.main:app \
--host 0.0.0.0 \
--port 8000 \
> api.log 2>&1 &
```

Check:

```bash
ps aux | grep uvicorn
```

However, for a more reliable service, use a process manager such as `systemd`.

---

### 10.12 Step 11: Run FastAPI with systemd

Create:

```text
/etc/systemd/system/ml-api.service
```

Example:

```ini
[Unit]
Description=ML FastAPI Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ml-deployment-project
Environment="PATH=/home/ubuntu/ml-deployment-project/.venv/bin"
ExecStart=/home/ubuntu/ml-deployment-project/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Adjust the user and paths for your EC2 environment.

Reload systemd:

```bash
sudo systemctl daemon-reload
```

Start:

```bash
sudo systemctl start ml-api
```

Enable at boot:

```bash
sudo systemctl enable ml-api
```

Check:

```bash
sudo systemctl status ml-api
```

View logs:

```bash
sudo journalctl -u ml-api -f
```

Now the service can automatically restart and start after a server reboot.

---

### 10.13 Step 12: Production-style architecture with Nginx

A better architecture is:

```text
                 Internet
                    |
                 HTTPS
                    |
                    v
              +-----------+
              |   Nginx   |
              +-----------+
                    |
                    v
              +-----------+
              |  FastAPI  |
              +-----------+
                    |
                    v
              +-----------+
              | ML Model  |
              +-----------+
```

Nginx can act as a reverse proxy.

The public user does not need to access:

```text
:8000
```

directly.

Instead:

```text
https://your-domain.com
```

can route internally to:

```text
http://127.0.0.1:8000
```

For a real public service, configure HTTPS using an appropriate certificate solution.

---

### 10.14 Step 13: Deploy Streamlit separately

If you want both:

```text
Streamlit UI
       |
       v
FastAPI
       |
       v
ML Model
```

you can run Streamlit separately.

Example:

```bash
streamlit run streamlit_app/app.py \
--server.address 0.0.0.0 \
--server.port 8501
```

However, for production, consider whether Streamlit should be public or whether another frontend is more appropriate.

A cleaner architecture can be:

```text
                    Internet
                       |
              +--------+--------+
              |                 |
              v                 v
          Frontend          ML API
                              |
                              v
                           Model
```

---

## 11. Production Checklist

This section provides a readiness review across the model, API, security, infrastructure, and delivery pipeline. Use it to identify operational gaps before exposing an inference service to real users.

Before calling an ML API production-ready, check the following.

### 11.1 Model

- [ ] Model is versioned.
- [ ] Preprocessing is included.
- [ ] Input features are validated.
- [ ] Model artifact comes from a trusted source.
- [ ] Model version is recorded.
- [ ] Training and inference environments are compatible.

### 11.2 API

- [ ] `/health` endpoint exists.
- [ ] Input validation is implemented.
- [ ] Meaningful HTTP status codes are returned.
- [ ] Exceptions are handled appropriately.
- [ ] Request timeouts are considered.
- [ ] Logging is implemented.
- [ ] API documentation is available.
- [ ] Authentication is added if required.

### 11.3 Security

- [ ] No passwords in source code.
- [ ] No AWS access keys in Git.
- [ ] Secrets are stored securely.
- [ ] SSH access is restricted.
- [ ] Only required ports are exposed.
- [ ] HTTPS is configured for public production traffic.

### 11.4 Infrastructure

- [ ] Application runs under a process manager.
- [ ] Application restarts automatically.
- [ ] Logs are available.
- [ ] CPU and memory are monitored.
- [ ] Disk usage is monitored.
- [ ] Backups are considered where required.

### 11.5 CI/CD

A mature workflow can be:

```text
Developer
    |
    v
Git commit
    |
    v
GitHub
    |
    v
CI Tests
    |
    v
Build
    |
    v
Deploy
    |
    v
AWS
```

---

## 12. Common Deployment Issues

This section maps frequent deployment symptoms to likely causes and first checks. It is intended as a focused troubleshooting reference for differences between notebook, local-service, and cloud environments.

### Issue 1: Model works in notebook but fails in API

Possible reasons:

- Different feature order.
- Missing preprocessing.
- Different library versions.
- Missing model artifact.
- Wrong working directory.

Solution:

Use a Pipeline and define a clear input schema.

---

### Issue 2: `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'sklearn'
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Also confirm that the virtual environment is activated.

---

### Issue 3: API works locally but not on EC2

Check:

```text
1. Application host
2. EC2 security group
3. Linux firewall
4. Correct port
5. Process status
6. EC2 public IP
```

For external access, the server should normally listen on:

```text
0.0.0.0
```

rather than only:

```text
127.0.0.1
```

---

### Issue 4: Model file not found

Example:

```text
FileNotFoundError
```

Avoid fragile relative paths where possible.

A better approach is to build paths relative to the project/module location.

Example:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "model_pipeline.pkl"
)
```

Then:

```python
model = joblib.load(MODEL_PATH)
```

---

### Issue 5: API stops when terminal closes

If you started:

```bash
uvicorn api.main:app
```

inside an SSH session, the process may stop when the session ends.

Use a proper process manager such as `systemd` for a persistent service.

---

### Issue 6: Streamlit cannot connect to FastAPI

Check:

```text
FastAPI running?
Correct API URL?
Correct port?
Security group configured?
CORS required?
Network route available?
```

Test the FastAPI endpoint independently before debugging Streamlit.

---

## 13. End-to-End Workflow

This section consolidates the preceding material into a single lifecycle, from model development through cloud hosting. It provides a compact reference for understanding how artifacts, application components, and infrastructure fit together.

The complete workflow can be summarized as follows.

### Phase 1: Model Development

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Evaluation
```

### Phase 2: Model Packaging

```text
Best Model
    ↓
Pipeline
    ↓
joblib
    ↓
model_pipeline.pkl
```

### Phase 3: Application Development

```text
model_pipeline.pkl
       |
       +------> Prediction Script
       |
       +------> Streamlit
       |
       +------> FastAPI
```

### Phase 4: Integration

```text
Streamlit
    |
    | HTTP POST
    v
FastAPI
    |
    v
Prediction Function
    |
    v
ML Pipeline
```

### Phase 5: Version Control

```text
Local Project
     ↓
Git
     ↓
GitHub
```

### Phase 6: Cloud Deployment

```text
GitHub
   ↓
AWS EC2
   ↓
Virtual Environment
   ↓
Dependencies
   ↓
FastAPI
   ↓
ML Model
```

### Phase 7: Production Architecture

```text
                    User
                     |
                     v
                HTTPS / DNS
                     |
                     v
                  Nginx
                     |
                     v
                  FastAPI
                     |
             +-------+-------+
             |               |
             v               v
          Model          Monitoring
             |
             v
         Prediction
```

---

## Practical Project: Recommended Final Structure

For a portfolio-ready ML deployment project, use:

```text
ml-deployment-project/
│
├── data/
│   └── README.md
│
├── models/
│   └── model_pipeline.pkl
│
├── notebooks/
│   └── model_experiment.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Final Takeaway

ML deployment is not simply:

```text
Train model → Save model
```

A practical deployment lifecycle is:

```text
                 ML Development
                       |
                       v
                Train & Evaluate
                       |
                       v
              Save Model Pipeline
                       |
                       v
              Python Prediction Code
                       |
            +----------+----------+
            |                     |
            v                     v
       Streamlit              FastAPI
            |                     |
            |                     |
            +----------+----------+
                       |
                       v
                   GitHub
                       |
                       v
                    AWS EC2
                       |
                       v
             Production ML API
                       |
                       v
              Monitoring & Updates
```

The key concept is to keep **training, prediction, API, UI, and infrastructure responsibilities separate**.

For a beginner-to-intermediate ML Engineer portfolio, a strong project demonstrates:

1. Model training.
2. Model serialization.
3. Reusable prediction code.
4. Streamlit UI.
5. FastAPI inference service.
6. API/UI integration.
7. Git/GitHub version control.
8. Linux deployment.
9. AWS EC2 deployment.
10. Basic production practices such as health checks, logging, security, and process management.

This progression gives you a practical foundation for moving from data-science experimentation to ML engineering and production deployment.
