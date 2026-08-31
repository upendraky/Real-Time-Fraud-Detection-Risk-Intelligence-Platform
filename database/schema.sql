CREATE TABLE IF NOT EXISTS fraud_predictions (
    id BIGSERIAL PRIMARY KEY,

    step INTEGER NOT NULL,
    transaction_type VARCHAR(30) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,

    origin_account VARCHAR(100),
    destination_account VARCHAR(100),

    old_balance_origin DOUBLE PRECISION,
    new_balance_origin DOUBLE PRECISION,

    old_balance_destination DOUBLE PRECISION,
    new_balance_destination DOUBLE PRECISION,

    fraud_probability DOUBLE PRECISION NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,

    is_fraud INTEGER NOT NULL,
    risk_level VARCHAR(20) NOT NULL,

    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
