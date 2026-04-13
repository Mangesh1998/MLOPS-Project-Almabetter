<h1 align="center">✈️ Travel Analytics MLOps Ecosystem</h1>

<div align="center">
  A comprehensive, multi-service machine learning deployment designed to orchestrate complex travel logistics algorithms seamlessly.
</div>

<br />

## 🌟 Overview

The **Travel Analytics Ecosystem** encompasses three distinctly specialized, scalable machine learning domains working harmoniously to address flight pricing, predictive demographics, and collaborative filtering for hotel selections. Developed with deployment resilience in mind, this project wraps predictive mathematical models into lightweight **Flask Web APIs**, coupled with visually premium and reactive **Streamlit Front-End Dashboards**.

---

## 🛠 Features & Architecture

This repository is bifurcated into three parallel AI subsystems:

### 1. Flight Price Predictor (`Travel_ML_System`)
- **Objective:** Dynamically calculates ticket costs relying on destination matrices, date sequencing, and corporate agency algorithms.
- **Modeling:** Employs a robust **RandomForestRegressor** native backend trained alongside strict mapping features.
- **Deployment:** Streamlit Dashboard on port `8501`. Linked Flask API logic running independently on `8000`.

### 2. Demographic Authentication Engine (`Gender Classification Model`)
- **Objective:** Leverages Natural Language Processing over algorithmic user arrays to perform semantic gender classifications.
- **Modeling:** Utilizes **SentenceTransformers (Flax MiniLM)** embedded vectors processed through **Principal Component Analysis (PCA)** prior to passing through a tuned classification network.
- **Deployment:** Streamlit Interface on `8502`. Flask RESTful pipeline running on `8001`.

### 3. Hotel Recommender Matrix (`Hotel Recommender System`)
- **Objective:** Cross-references travel history to formulate curated hotel selections.
- **Modeling:** Calculates dense metric relationships utilizing Sparse Matrix Factorization mapped over behavioral analytics databases.
- **Deployment:** Streamlit Portfolio Render operating on `8503`.

---

## ⚙️ Tech Stack

<details>
  <summary>Click to view system dependencies</summary>
  
- **Python**: 3.11.x
- **Core ML Frameworks**: `scikit-learn`, `SentenceTransformers`, `pandas`, `numpy`
- **Front-End Interfaces**: `streamlit`
- **Back-End Orchestration**: `flask`
- **Containerization**: `docker` (optional mapping available)

</details>

---

## ⚙️ Advanced MLOps Integrations

- **🌐 REST API Development:** Developed a Flask-based API to serve flight price predictions in real time.
- **📦 Containerization with Docker:** Packaged the model and API for portability and ease of deployment.
- **📈 Kubernetes Deployment:** Ensured scalability and efficient load management by deploying the Dockerized application using Kubernetes.
- **🔄 Workflow Automation with Apache Airflow:** Designed and implemented DAGs to automate data preprocessing and model training workflows.
- **🚀 CI/CD Pipeline with Jenkins:** Automated the integration and deployment process using a Jenkins pipeline, ensuring reliable and consistent releases.
- **🧬 Model Tracking with MLFlow:** Managed model versions systematically and tracked performance metrics during iterations.

---

## 🚀 Local Installation & Orchestration

> **Note:** The included predictive model (`.pkl`) binaries rely on native OS extraction processes. We have provided an automation script to effortlessly synthesize them to bypass remote object storage.

### 1. Environment Initialization
First, clone the project locally and download prerequisite framework distributions:
```bash
git clone https://github.com/Mangesh1998/MLOPS-Project-Almabetter.git
cd MLOPS-Project-Almabetter
python -m venv venv

# Activate Environment (Windows/PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements_all.txt
```

### 2. Executing the Environment Sequence
To safely bypass legacy execution parameters and natively bind the five application layers, run our master PowerShell orchestrator:
```bash
.\run_all.ps1
```
*(This gracefully isolates each interface and allocates execution arrays strictly matched to local file origins.)*

**The Ecosystem will instantly spin up:**
- Flight Prediction: [http://localhost:8501](http://localhost:8501)
- Gender Matrix: [http://localhost:8502](http://localhost:8502)
- Recommender Space: [http://localhost:8503](http://localhost:8503)

### 3. Custom Model Compilation
If desired, directly formulate your own predictive models mapped exactly to UI dimensions:
```bash
python train_models.py
```

---

<div align="center">
  <i>Maintained and tailored natively for rapid local web ecosystem execution.</i>
</div>
