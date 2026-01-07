```markdown
#  Retail Intelligence & Churn MLOps Pipeline
![MLOps Pipeline](https://github.com/ernestekaitoo/End-to-end-retail-projects/actions/workflows/mlops-pipeline.yml/badge.svg)

An end-to-end MLOps system that processes **1.06M+ retail transactions** to predict customer churn and forecast sales. This project demonstrates a production-grade architecture using microservices and automated CI/CD.

##  System Architecture
The system is built using a **Decoupled Microservices Architecture** orchestrated with Docker Compose:

* **The Brain (Backend):** A FastAPI service serving a Random Forest classifier and a Prophet forecasting engine.
* **The Face (Frontend):** An interactive Streamlit dashboard for real-time customer risk assessment.
* **The Pipeline:** Automated data cleaning, RFM analysis, and K-Means clustering.



##  Tech Stack
* **Data:** Pandas, NumPy (Processing 1.06M rows)
* **ML/AI:** Scikit-Learn (Random Forest), Prophet (Forecasting)
* **API/UI:** FastAPI, Streamlit
* **DevOps:** Docker, Docker Compose, GitHub Actions (CI/CD)

##  How to Run
Ensure you have Docker Desktop running, then use the provided Makefile:

```bash
# Build and launch the entire stack
make up

```

Access the **Dashboard** at `http://localhost:8501` and the **API Docs** at `http://localhost:8000/docs`.

##  Key Achievements

* **Orchestrated Microservices:** Decoupled model inference from the UI for improved scalability.
* **Automated CI/CD:** Integrated GitHub Actions to automate linting and Docker builds.
* **Scale:** Successfully handled a global retail dataset with over 1 million records.

```

---

## 2. The LinkedIn Announcement
Post this along with a screenshot of your **Streamlit Dashboard** or a screen-recording of the **GitHub Actions** green checkmarks.

**Headline:**  Just launched: A Production-Grade MLOps Pipeline for Retail Intelligence

**Body:**
I’m excited to share my latest project—a full-stack MLOps system designed to handle the complexities of real-world retail data. 

Moving beyond notebooks, I engineered a decoupled **Microservices Architecture** that processes over **1.06 million transactions** to identify at-risk customers and forecast future revenue.

**Key Technical Highlights:**
   **Containerization:** Orchestrated a dual-container environment using **Docker Compose** (FastAPI backend + Streamlit frontend).
   **CI/CD:** Implemented a **GitHub Actions** pipeline to automate environment builds and code quality checks.
   **ML Engineering:** Developed a Random Forest classifier and a Prophet forecasting model, ensuring 100% feature parity via **Joblib** serialization.

This project taught me that "clean code" is just the start—building a reproducible, scalable infrastructure is what brings ML to life.

Check out the repo here: [https://github.com/ernestekaitoo/End-to-end-retail-projects]

#MLOps #DataEngineering #Docker #FastAPI #DataScience #MachineLearning

---

