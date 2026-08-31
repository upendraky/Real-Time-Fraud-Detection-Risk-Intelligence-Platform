import json
import time

import pandas as pd
from kafka import KafkaProducer


KAFKA_SERVER = "localhost:9092"
TOPIC = "fraud-transactions"

DATA_PATH = "data/processed/test.csv"

TRANSACTIONS_TO_SEND = 20
DELAY_SECONDS = 0.5


producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)


df = pd.read_csv(DATA_PATH)

fraud_df = df[df["isFraud"] == 1]
legitimate_df = df[df["isFraud"] == 0]

sample_size = TRANSACTIONS_TO_SEND // 2

sample_fraud = fraud_df.sample(
    n=sample_size,
    random_state=42
)

sample_legitimate = legitimate_df.sample(
    n=sample_size,
    random_state=42
)

sample = pd.concat(
    [sample_fraud, sample_legitimate]
).sample(
    frac=1,
    random_state=42
)


print(
    f"Sending {len(sample)} transactions "
    f"to Kafka topic '{TOPIC}'..."
)


for _, row in sample.iterrows():

    transaction = row.drop(
        labels=["isFraud"]
    ).to_dict()

    transaction = {
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in transaction.items()
    }

    producer.send(
        TOPIC,
        value=transaction
    )

    print(
        f"Sent transaction | "
        f"type={transaction['type']} | "
        f"amount={transaction['amount']:.2f}"
    )

    time.sleep(DELAY_SECONDS)


producer.flush()
producer.close()

print("Finished sending transactions.")
