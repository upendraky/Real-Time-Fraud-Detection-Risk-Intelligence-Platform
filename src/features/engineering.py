import numpy as np
import pandas as pd


ID_COLUMNS = ["nameOrig", "nameDest"]
TARGET = "isFraud"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model features from raw PaySim transactions.
    """

    data = df.copy()

    # Preserve IDs only for external use; never feed them to the model
    data = data.drop(
        columns=ID_COLUMNS,
        errors="ignore"
    )

    # Balance movement
    data["orig_balance_change"] = (
        data["oldbalanceOrg"] -
        data["newbalanceOrig"]
    )

    data["dest_balance_change"] = (
        data["newbalanceDest"] -
        data["oldbalanceDest"]
    )

    # Amount consistency
    data["orig_amount_gap"] = (
        data["orig_balance_change"] -
        data["amount"]
    )

    data["dest_amount_gap"] = (
        data["dest_balance_change"] -
        data["amount"]
    )

    # Ratios
    epsilon = 1e-6

    data["amount_to_orig_balance"] = (
        data["amount"] /
        (data["oldbalanceOrg"] + epsilon)
    )

    data["amount_to_dest_balance"] = (
        data["amount"] /
        (data["oldbalanceDest"] + epsilon)
    )

    # Zero-balance indicators
    data["orig_balance_zero_before"] = (
        data["oldbalanceOrg"] == 0
    ).astype(int)

    data["orig_balance_zero_after"] = (
        data["newbalanceOrig"] == 0
    ).astype(int)

    data["dest_balance_zero_before"] = (
        data["oldbalanceDest"] == 0
    ).astype(int)

    data["dest_balance_zero_after"] = (
        data["newbalanceDest"] == 0
    ).astype(int)

    # Amount transformation
    data["log_amount"] = np.log1p(
        data["amount"]
    )

    return data