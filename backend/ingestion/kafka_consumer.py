"""
backend/ingestion/kafka_consumer.py

Stub for a future streaming ingestion path. NOT active in Layer 1 — the
hackathon pipeline is batch-only (CSV -> Polars -> Parquet -> Postgres).
This exists so the architecture story ("stream-ready") holds up, and so a
real consumer can be dropped in later without touching batch_loader.py.

To activate: pip install confluent-kafka (not in requirements.txt yet),
set KAFKA_BOOTSTRAP_SERVERS, and call `consume_loop()` from a long-running
process (separate from the FastAPI app).
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger("btip.kafka_consumer")

TOPIC = "btip.violations.raw"


def handle_message(payload: dict) -> None:
    """Placeholder handler — would validate + upsert into `violations` the
    same way batch_loader/validator/seed_db do today."""
    logger.info("Received message on %s: %s", TOPIC, json.dumps(payload)[:200])


def consume_loop(bootstrap_servers: str | None = None) -> None:
    raise NotImplementedError(
        "Kafka streaming is not wired up yet. This is a placeholder for "
        "post-hackathon productionization. Use batch_loader.py for now."
    )


if __name__ == "__main__":
    print("kafka_consumer.py is a stub — no-op. Use batch_loader.py for ingestion.")
