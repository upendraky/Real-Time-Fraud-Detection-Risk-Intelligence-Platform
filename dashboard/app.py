import time

import requests
import streamlit as st


# ==================================================
# Configuration
# ==================================================

API_URL = "http://127.0.0.1:8000"


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Fraud Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
)


# ==================================================
# Helper Functions
# ==================================================

def get_api(endpoint):

    try:

        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:

        st.error(
            f"API connection failed: {e}"
        )

        return None


# ==================================================
# Header
# ==================================================

st.title(
    "🛡️ Real-Time Fraud Risk Intelligence"
)

st.caption(
    "Kafka + XGBoost + PostgreSQL + Redis + FastAPI"
)


# ==================================================
# System Health
# ==================================================

health = get_api("/health")

if health:

    st.success(
        "Fraud Detection System Online"
    )

else:

    st.error(
        "Fraud Detection API unavailable"
    )


# ==================================================
# Statistics
# ==================================================

stats = get_api("/stats")


if stats:

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            stats["total_transactions"],
        )

    with col2:

        st.metric(
            "Fraud Transactions",
            stats["fraud_transactions"],
        )

    with col3:

        st.metric(
            "Fraud Rate",
            f"{stats['fraud_rate'] * 100:.2f}%",
        )

    with col4:

        st.metric(
            "Average Risk Score",
            f"{stats['average_risk_score']:.2f}",
        )


st.divider()


# ==================================================
# Recent Fraud Alerts
# ==================================================

st.subheader(
    "🚨 Recent Fraud Alerts"
)

alerts = get_api(
    "/fraud/alerts?limit=10"
)


if alerts and alerts["alerts"]:

    alert_rows = []

    for alert in alerts["alerts"]:

        alert_rows.append({

            "ID": alert["id"],

            "Type": alert["transaction_type"],

            "Amount": alert["amount"],

            "Origin": alert["origin_account"],

            "Destination": alert[
                "destination_account"
            ],

            "Fraud Probability": round(
                alert["fraud_probability"],
                6,
            ),

            "Risk Score": round(
                alert["risk_score"],
                2,
            ),

            "Risk": alert["risk_level"],

        })

    st.dataframe(
        alert_rows,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No fraud alerts found."
    )


st.divider()


# ==================================================
# Recent Transactions
# ==================================================

st.subheader(
    "📊 Recent Transactions"
)

transactions = get_api(
    "/transactions/recent?limit=20"
)


if transactions:

    transaction_rows = []

    for transaction in transactions[
        "transactions"
    ]:

        transaction_rows.append({

            "ID": transaction["id"],

            "Step": transaction["step"],

            "Type": transaction[
                "transaction_type"
            ],

            "Amount": transaction[
                "amount"
            ],

            "Fraud Probability": transaction[
                "fraud_probability"
            ],

            "Risk Score": transaction[
                "risk_score"
            ],

            "Prediction": (
                "FRAUD"
                if transaction["is_fraud"]
                else "LEGITIMATE"
            ),

            "Risk": transaction[
                "risk_level"
            ],

        })

    st.dataframe(
        transaction_rows,
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# ==================================================
# Account Risk Lookup
# ==================================================

st.subheader(
    "🔍 Account Risk Lookup"
)

account_id = st.text_input(
    "Enter Origin Account ID",
    placeholder="Example: C1316763794",
)


if st.button(
    "Check Risk",
    type="primary",
):

    if not account_id:

        st.warning(
            "Enter an account ID."
        )

    else:

        risk = get_api(
            f"/risk/{account_id}"
        )

        if risk:

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Risk Level",
                    risk.get(
                        "risk_level",
                        "UNKNOWN",
                    ),
                )

            with col2:

                st.metric(
                    "Risk Score",
                    round(
                        float(
                            risk.get(
                                "risk_score",
                                0,
                            )
                        ),
                        2,
                    ),
                )

            with col3:

                st.metric(
                    "Fraud Probability",
                    f"{float(risk.get('fraud_probability', 0)) * 100:.2f}%",
                )

            st.json(risk)


st.divider()


# ==================================================
# Footer
# ==================================================

st.caption(
    "Real-time fraud detection powered by XGBoost"
)
