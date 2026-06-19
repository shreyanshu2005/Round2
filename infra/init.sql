-- Runs once on first container start (docker-entrypoint-initdb.d)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS violations (
    id                  TEXT PRIMARY KEY,
    latitude            DOUBLE PRECISION NOT NULL,
    longitude           DOUBLE PRECISION NOT NULL,
    geom                GEOMETRY(POINT, 4326),
    location            TEXT,
    vehicle_number      TEXT,
    vehicle_type        TEXT,
    violation_type      TEXT,        -- raw JSON-array-as-string, e.g. ["WRONG PARKING"]
    offence_code        TEXT,        -- raw JSON-array-as-string, e.g. [112,104]
    created_datetime    TIMESTAMPTZ NOT NULL,
    closed_datetime     TIMESTAMPTZ,
    modified_datetime   TIMESTAMPTZ,
    device_id           TEXT,
    created_by_id       TEXT,        -- closest proxy to "officer_id"
    center_code         DOUBLE PRECISION,
    police_station      TEXT,
    data_sent_to_scita  BOOLEAN,
    junction_name       TEXT,        -- closest proxy to "junction_id" (free text, e.g. "BTP211 - Central Street Junction" / "No Junction")
    action_taken_timestamp        TIMESTAMPTZ,
    data_sent_to_scita_timestamp  TIMESTAMPTZ,
    updated_vehicle_number TEXT,
    updated_vehicle_type   TEXT,
    validation_status      TEXT,     -- approved / rejected / created1 / null
    validation_timestamp   TIMESTAMPTZ
);

-- TimescaleDB hypertable partitioned on created_datetime (the only reliably-populated timestamp)
SELECT create_hypertable('violations', 'created_datetime', if_not_exists => TRUE, migrate_data => TRUE);

CREATE INDEX IF NOT EXISTS idx_violations_geom ON violations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_violations_junction ON violations (junction_name);
CREATE INDEX IF NOT EXISTS idx_violations_created ON violations (created_datetime DESC);
