# MLOps Confidence Collapse Monitoring System

## Overview

This project is an end-to-end MLOps pipeline designed to study and monitor model confidence behavior under noisy input conditions. It demonstrates a full production-grade machine learning lifecycle including experiment tracking, model deployment, containerization, CI/CD automation, and real-time observability.

The system is built to simulate real-world ML deployment challenges such as performance degradation under input noise and provides monitoring capabilities to observe model behavior using Prometheus and Grafana.

---

## Problem Statement

Machine learning models often perform well during training but degrade unpredictably in production due to input noise, distribution shifts, and data inconsistencies. Most systems fail to provide visibility into these failures, making debugging and analysis difficult.

This project addresses that gap by building an observable ML system that tracks model confidence changes under varying input noise levels and provides real-time monitoring through a complete MLOps stack.

---

## Key Features

- End-to-end ML pipeline with FastAPI inference service
- Experiment tracking using MLflow
- Containerized deployment using Docker and Docker Compose
- Real-time monitoring using Prometheus
- Visualization dashboards using Grafana
- CI/CD pipeline using Jenkins
- Load testing for simulated production traffic
- Confidence tracking under input noise conditions

---

## Tech Stack

### Machine Learning
- Scikit-learn
- NumPy
- MLflow

### Backend
- FastAPI
- Uvicorn

### MLOps & Deployment
- Docker
- Docker Compose
- Jenkins (CI/CD)

### Monitoring & Observability
- Prometheus
- Grafana

---

## System Architecture

The system follows a modular MLOps architecture:

1. Client sends input features to FastAPI service
2. API processes input and runs ML model inference
3. Prediction and confidence scores are returned
4. Metrics are exposed via `/metrics` endpoint
5. Prometheus scrapes metrics at regular intervals
6. Grafana visualizes system performance and model behavior
7. MLflow tracks experiments and model versions
8. Jenkins automates build and deployment pipeline

---

## How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/ali-ab-2003/mlops-confidence-projects.git
cd mlops-confidence-projects
