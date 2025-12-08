# MLOps Pipeline Project – Classification with Version Control, Tracking, Explainability & Web Interface

## Introduction
This project's goal is to build a complete **MLOps pipeline** for a classification task, integrating:
- **Version control** (Git & DVC)
- **Experiment tracking** (MLflow)
- **Model explainability** (SHAP/LIME)
- **Interactive web interface** (Streamlit)

---

## Get started

Create virtual environment : 
- <code>$ python -m venv venv</code>
- <code>$ source venv/bin/actiate</code>
Install required libraries :
- <code>$ pip install -r requirements.txt</code>
Pull data from DVC :
- <code>$ dvc pull</code>
Go to the <code>notebooks</code> folder :
1. Look through <code>01_exploration.ipynb</code> file.
2. Execute all cells in <code>03_experiments.ipynb</code> file.
3. In a terminal open in this root folder, type <code>./run_mlflow.bat</code> to open MLFlow UI
4. In UI, look through the 3 first experiments.
5. In <code>03_explainability.ipynb</code> file, look through the cells.
6. In a terminal, run the command <code>streamlit run streamlit_sipkamed.py</code>

## Team
* Amine ELHANA
* Sarah LAKHZOUM
* Abdelbadie Dhiaeddine NESSAS

---

## Dataset
- SIPaKMeD dataset (isolated cells for cervical cancer)
- Dataset versioning is managed using **DVC**.

---

## Tools used
- **Git** – Code version control
- **DVC** – Data & model versioning
- **MLflow** – Experiment tracking (≥10 experiments)
- **SHAP / LIME** – Model explainability
- **Streamlit / Gradio** – Web interface

---