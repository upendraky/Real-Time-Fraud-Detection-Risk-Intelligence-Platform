# Real-Time Fraud Detection & Risk Intelligence Platform

An end-to-end real-time fraud detection system built using the PaySim financial transaction dataset.

The platform uses machine learning to identify potentially fraudulent transactions and assign a real-time risk score. XGBoost is used as the primary fraud detection model, with SHAP used to provide model explainability.

## Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- SHAP
- Apache Kafka
- PostgreSQL
- Redis
- FastAPI
- Streamlit
- Docker & Docker Compose

## Key Features

- Transaction data analysis and feature engineering
- Fraud detection using XGBoost
- Fraud probability and risk scoring
- Optimized classification threshold
- SHAP-based model explainability
- Real-time transaction streaming with Kafka
- Prediction storage in PostgreSQL
- Live risk state using Redis
- REST API using FastAPI
- Interactive Streamlit dashboard
- Containerized deployment using Docker Compose

## Model Performance

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.994000 | 0.753838 |
| Random Forest | 0.999704 | 0.967007 |
| XGBoost | **0.999913** | **0.987490** |

## Architecture

PaySim → Kafka → ML Consumer → XGBoost → PostgreSQL / Redis → FastAPI → Streamlit

## Project Goal

The goal of this project is to demonstrate how a machine learning fraud detection model can be integrated into a production-oriented real-time data pipeline for transaction monitoring and risk intelligence.

## Status

Completed — end-to-end fraud detection, real-time streaming, API, database, caching, dashboard, and Docker deployment.
