import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import redis

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)
from pydantic import BaseModel


# ==================================================
# Configuration
# ==================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fraud_model.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "preprocessor.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fraud_threshold.json"
)


# ==================================================
# Environment
# ==================================================

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud_detection")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_password")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None


# ==================================================
# FastAPI
# ==================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "Fraud detection and risk "
        "intelligence using XGBoost"
    ),
    version="1.0.0",
)


# ==================================================
# Load Model
# ==================================================

model = joblib.load(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

with open(
    THRESHOLD_PATH,
    "r",
) as f:

    threshold_config = __import__(
        "json"
    ).load(f)

FRAUD_THRESHOLD = threshold_config[
    "threshold"
]


# ==================================================
# PostgreSQL
# ==================================================


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        connect_timeout=10,
    )



# ==================================================
# Redis
# ==================================================


redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=0,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)



# ==================================================
# Transaction Schema
# ==================================================

class Transaction(BaseModel):

    step: int

    type: str

    amount: float

    nameOrig: str

    oldbalanceOrg: float

    newbalanceOrig: float

    nameDest: str

    oldbalanceDest: float

    newbalanceDest: float

    isFlaggedFraud: int = 0


# ==================================================
# Feature Engineering
# ==================================================

def create_features(
    transaction
):

    df = pd.DataFrame(
        [transaction]
    )

    df = df.drop(
        columns=[
            "nameOrig",
            "nameDest",
        ],
        errors="ignore",
    )

    df["orig_balance_change"] = (
        df["oldbalanceOrg"]
        - df["newbalanceOrig"]
    )

    df["dest_balance_change"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

    df["orig_amount_gap"] = (
        df["orig_balance_change"]
        - df["amount"]
    )

    df["dest_amount_gap"] = (
        df["dest_balance_change"]
        - df["amount"]
    )

    epsilon = 1e-6

    df["amount_to_orig_balance"] = (
        df["amount"]
        / (
            df["oldbalanceOrg"]
            + epsilon
        )
    )

    df["amount_to_dest_balance"] = (
        df["amount"]
        / (
            df["oldbalanceDest"]
            + epsilon
        )
    )

    df["orig_balance_zero_before"] = (
        df["oldbalanceOrg"] == 0
    ).astype(int)

    df["orig_balance_zero_after"] = (
        df["newbalanceOrig"] == 0
    ).astype(int)

    df["dest_balance_zero_before"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    df["dest_balance_zero_after"] = (
        df["newbalanceDest"] == 0
    ).astype(int)

    df["log_amount"] = np.log1p(
        df["amount"]
    )

    return df


# ==================================================
# Risk
# ==================================================

def get_risk_level(
    probability
):

    if probability >= 0.80:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


# ==================================================
# Health
# ==================================================

@app.get("/health")
def health():

    postgres_status = "healthy"

    redis_status = "healthy"

    try:

        connection = get_db_connection()

        connection.close()

    except Exception:

        postgres_status = "unhealthy"

    try:

        redis_client.ping()

    except Exception:

        redis_status = "unhealthy"

    return {
        "status": (
            "healthy"
            if (
                postgres_status == "healthy"
                and redis_status == "healthy"
            )
            else "degraded"
        ),
        "model": "XGBoost",
        "threshold": FRAUD_THRESHOLD,
        "postgresql": postgres_status,
        "redis": redis_status,
    }


# ==================================================
# Prediction
# ==================================================

@app.post("/predict")
def predict(
    transaction: Transaction
):

    transaction_dict = (
        transaction.model_dump()
    )

    features = create_features(
        transaction_dict
    )

    processed_features = (
        preprocessor.transform(
            features
        )
    )

    fraud_probability = float(
        model.predict_proba(
            processed_features
        )[0, 1]
    )

    is_fraud = int(
        fraud_probability
        >= FRAUD_THRESHOLD
    )

    risk_score = (
        fraud_probability * 100
    )

    risk_level = get_risk_level(
        fraud_probability
    )

    return {
        "fraud_probability": round(
            fraud_probability,
            6,
        ),
        "risk_score": round(
            risk_score,
            2,
        ),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "threshold": FRAUD_THRESHOLD,
    }


# ==================================================
# Account Risk
# ==================================================

@app.get("/risk/{account_id}")
def get_account_risk(
    account_id: str
):

    key = f"risk:{account_id}"

    risk_data = redis_client.hgetall(
        key
    )

    if not risk_data:

        raise HTTPException(
            status_code=404,
            detail=(
                "No recent risk data "
                "found for this account"
            ),
        )

    return risk_data


# ==================================================
# Recent Transactions
# ==================================================

@app.get("/transactions/recent")
def recent_transactions(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    )
):

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            step,
            transaction_type,
            amount,
            origin_account,
            destination_account,
            fraud_probability,
            risk_score,
            is_fraud,
            risk_level,
            processed_at
        FROM fraud_predictions
        ORDER BY processed_at DESC
        LIMIT %s
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    cursor.close()

    connection.close()

    transactions = []

    for row in rows:

        transactions.append({

            "id": row[0],

            "step": row[1],

            "transaction_type": row[2],

            "amount": row[3],

            "origin_account": row[4],

            "destination_account": row[5],

            "fraud_probability": row[6],

            "risk_score": row[7],

            "is_fraud": row[8],

            "risk_level": row[9],

            "processed_at": row[10],
        })

    return {
        "count": len(
            transactions
        ),
        "transactions": transactions,
    }


# ==================================================
# Fraud Alerts
# ==================================================

@app.get("/fraud/alerts")
def fraud_alerts(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    )
):

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            step,
            transaction_type,
            amount,
            origin_account,
            destination_account,
            fraud_probability,
            risk_score,
            risk_level,
            processed_at
        FROM fraud_predictions
        WHERE is_fraud = 1
        ORDER BY processed_at DESC
        LIMIT %s
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    cursor.close()

    connection.close()

    alerts = []

    for row in rows:

        alerts.append({

            "id": row[0],

            "step": row[1],

            "transaction_type": row[2],

            "amount": row[3],

            "origin_account": row[4],

            "destination_account": row[5],

            "fraud_probability": row[6],

            "risk_score": row[7],

            "risk_level": row[8],

            "processed_at": row[9],
        })

    return {
        "count": len(alerts),
        "alerts": alerts,
    }


# ==================================================
# Statistics
# ==================================================

@app.get("/stats")
def statistics():

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (
                WHERE is_fraud = 1
            ),
            COUNT(*) FILTER (
                WHERE is_fraud = 0
            ),
            AVG(fraud_probability),
            AVG(risk_score)
        FROM fraud_predictions
        """
    )

    row = cursor.fetchone()

    cursor.close()

    connection.close()

    total = row[0] or 0

    fraud = row[1] or 0

    legitimate = row[2] or 0

    fraud_rate = (
        fraud / total
        if total > 0
        else 0
    )

    return {

        "total_transactions": total,

        "fraud_transactions": fraud,

        "legitimate_transactions": legitimate,

        "fraud_rate": round(
            fraud_rate,
            6,
        ),

        "average_fraud_probability": round(
            float(row[3] or 0),
            6,
        ),

        "average_risk_score": round(
            float(row[4] or 0),
            2,
        ),
    }
