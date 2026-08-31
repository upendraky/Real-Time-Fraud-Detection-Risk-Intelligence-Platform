import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import redis
from kafka import KafkaConsumer


# ==================================================
# Logging
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("fraud-consumer")


# ==================================================
# Environment Configuration
# ==================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = int(
    os.getenv("POSTGRES_PORT", "5432")
)

REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "localhost",
)

REDIS_PORT = int(
    os.getenv("REDIS_PORT", "6379")
)


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

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
# Load ML Artifacts
# ==================================================

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

FRAUD_THRESHOLD = threshold_config["threshold"]


# ==================================================
# PostgreSQL
# ==================================================

DB_CONFIG = {
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
    "database": "fraud_detection",
    "user": "fraud_user",
    "password": "fraud_password",
}

db_connection = psycopg2.connect(
    **DB_CONFIG
)

db_cursor = db_connection.cursor()

logger.info("PostgreSQL connected")


# ==================================================
# Redis
# ==================================================

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True,
)

redis_client.ping()

logger.info("Redis connected")


# ==================================================
# Kafka
# ==================================================

def deserialize_transaction(value):
    return json.loads(
        value.decode("utf-8")
    )


consumer = KafkaConsumer(
    "fraud-transactions",
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-detection-db-consumer",
    value_deserializer=deserialize_transaction,
)


# ==================================================
# Feature Engineering
# ==================================================

def create_features(transaction):

    df = pd.DataFrame([transaction])

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
# Risk Classification
# ==================================================

def get_risk_level(probability):

    if probability >= 0.80:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


# ==================================================
# Save Prediction to PostgreSQL
# ==================================================

def save_prediction(
    transaction,
    fraud_probability,
    risk_score,
    is_fraud,
    risk_level,
):

    query = """
        INSERT INTO fraud_predictions (
            step,
            transaction_type,
            amount,
            origin_account,
            destination_account,
            old_balance_origin,
            new_balance_origin,
            old_balance_destination,
            new_balance_destination,
            fraud_probability,
            risk_score,
            is_fraud,
            risk_level
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
        )
    """

    values = (
        transaction["step"],
        transaction["type"],
        transaction["amount"],
        transaction["nameOrig"],
        transaction["nameDest"],
        transaction["oldbalanceOrg"],
        transaction["newbalanceOrig"],
        transaction["oldbalanceDest"],
        transaction["newbalanceDest"],
        fraud_probability,
        risk_score,
        is_fraud,
        risk_level,
    )

    db_cursor.execute(
        query,
        values,
    )

    db_connection.commit()


# ==================================================
# Save Prediction to Redis
# ==================================================

def save_to_redis(
    transaction,
    fraud_probability,
    risk_score,
    is_fraud,
    risk_level,
):

    origin_account = transaction["nameOrig"]

    redis_key = f"risk:{origin_account}"

    risk_data = {
        "step": str(transaction["step"]),
        "transaction_type": str(
            transaction["type"]
        ),
        "amount": str(
            transaction["amount"]
        ),
        "origin_account": str(
            transaction["nameOrig"]
        ),
        "destination_account": str(
            transaction["nameDest"]
        ),
        "fraud_probability": str(
            fraud_probability
        ),
        "risk_score": str(
            risk_score
        ),
        "is_fraud": str(
            is_fraud
        ),
        "risk_level": str(
            risk_level
        ),
    }

    redis_client.hset(
        redis_key,
        mapping=risk_data,
    )

    redis_client.expire(
        redis_key,
        3600,
    )


# ==================================================
# Main Consumer Loop
# ==================================================

logger.info(
    "Fraud Detection Consumer started"
)

logger.info(
    "Kafka bootstrap server: %s",
    KAFKA_BOOTSTRAP_SERVERS,
)

logger.info(
    "Waiting for new transactions"
)


try:

    for message in consumer:

        transaction = message.value

        try:

            # ------------------------------------------
            # Feature Engineering
            # ------------------------------------------

            features = create_features(
                transaction
            )

            # ------------------------------------------
            # Preprocessing
            # ------------------------------------------

            processed_features = (
                preprocessor.transform(
                    features
                )
            )

            # ------------------------------------------
            # Model Prediction
            # ------------------------------------------

            fraud_probability = float(
                model.predict_proba(
                    processed_features
                )[0, 1]
            )

            # ------------------------------------------
            # Fraud Decision
            # ------------------------------------------

            is_fraud = int(
                fraud_probability
                >= FRAUD_THRESHOLD
            )

            # ------------------------------------------
            # Risk Score
            # ------------------------------------------

            risk_score = (
                fraud_probability * 100
            )

            # ------------------------------------------
            # Risk Level
            # ------------------------------------------

            risk_level = get_risk_level(
                fraud_probability
            )

            # ------------------------------------------
            # PostgreSQL
            # ------------------------------------------

            save_prediction(
                transaction,
                fraud_probability,
                risk_score,
                is_fraud,
                risk_level,
            )

            # ------------------------------------------
            # Redis
            # ------------------------------------------

            save_to_redis(
                transaction,
                fraud_probability,
                risk_score,
                is_fraud,
                risk_level,
            )

            # ------------------------------------------
            # Console Output
            # ------------------------------------------

            print("=" * 60)

            print(
                f"Transaction: "
                f"{transaction['nameOrig']}"
            )

            print(
                f"Type: "
                f"{transaction['type']}"
            )

            print(
                f"Amount: "
                f"{transaction['amount']:.2f}"
            )

            print(
                f"Fraud Probability: "
                f"{fraud_probability:.6f}"
            )

            print(
                f"Risk Score: "
                f"{risk_score:.2f}"
            )

            print(
                f"Prediction: "
                f"{'FRAUD' if is_fraud else 'LEGITIMATE'}"
            )

            print(
                f"Risk Level: "
                f"{risk_level}"
            )

            print(
                "Saved to PostgreSQL: YES"
            )

            print(
                "Saved to Redis: YES"
            )

            logger.info(
                "Processed transaction=%s "
                "type=%s amount=%.2f "
                "probability=%.6f "
                "prediction=%s risk=%s",
                transaction["nameOrig"],
                transaction["type"],
                transaction["amount"],
                fraud_probability,
                (
                    "FRAUD"
                    if is_fraud
                    else "LEGITIMATE"
                ),
                risk_level,
            )

        except Exception:

            db_connection.rollback()

            logger.exception(
                "Error processing transaction"
            )


except KeyboardInterrupt:

    logger.info(
        "Consumer stopped"
    )


finally:

    db_cursor.close()

    db_connection.close()

    redis_client.close()

    consumer.close()

    logger.info(
        "Connections closed"
    )
