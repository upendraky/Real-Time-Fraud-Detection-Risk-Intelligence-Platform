import json
import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "data/processed/xgb_fraud_model.pkl"
THRESHOLD_PATH = "data/processed/fraud_threshold.json"


class FraudPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH
    ):
        self.model = joblib.load(model_path)

        with open(threshold_path, "r") as f:
            config = json.load(f)

        self.threshold = config["threshold"]

    def preprocess(self, transaction):
        df = pd.DataFrame([transaction])

        # Remove identifiers from model input
        df = df.drop(
            columns=["nameOrig", "nameDest"],
            errors="ignore"
        )

        # Encode transaction type
        df = pd.get_dummies(
            df,
            columns=["type"],
            dtype=int
        )

        # Add engineered features
        df["orig_balance_change"] = (
            df["oldbalanceOrg"] -
            df["newbalanceOrig"]
        )

        df["dest_balance_change"] = (
            df["newbalanceDest"] -
            df["oldbalanceDest"]
        )

        df["orig_amount_gap"] = (
            df["orig_balance_change"] -
            df["amount"]
        )

        df["dest_amount_gap"] = (
            df["dest_balance_change"] -
            df["amount"]
        )

        epsilon = 1e-6

        df["amount_to_orig_balance"] = (
            df["amount"] /
            (df["oldbalanceOrg"] + epsilon)
        )

        df["amount_to_dest_balance"] = (
            df["amount"] /
            (df["oldbalanceDest"] + epsilon)
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

    def predict(self, transaction):

        X = self.preprocess(transaction)

        probability = self.model.predict_proba(X)[0, 1]

        prediction = int(
            probability >= self.threshold
        )

        return {
            "fraud_probability": float(probability),
            "risk_score": float(probability * 100),
            "is_fraud": prediction,
            "threshold": float(self.threshold)
        }