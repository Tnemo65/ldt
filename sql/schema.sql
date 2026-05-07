-- sql/schema.sql
-- PostgreSQL schema for CA-DQStream pipeline
-- Spec: Section 2.3, Lines 250-275

CREATE TABLE IF NOT EXISTS taxi_trips_raw (
    trip_id VARCHAR(64) PRIMARY KEY,
    vendor_id INTEGER,
    pickup_datetime TIMESTAMP NOT NULL,
    dropoff_datetime TIMESTAMP NOT NULL,
    passenger_count INTEGER,
    trip_distance DOUBLE PRECISION,
    pickup_location_id INTEGER,
    dropoff_location_id INTEGER,
    payment_type INTEGER,
    fare_amount DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pickup_datetime ON taxi_trips_raw(pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_pickup_location ON taxi_trips_raw(pickup_location_id);

CREATE TABLE IF NOT EXISTS schema_violations (
    violation_id SERIAL PRIMARY KEY,
    trip_id VARCHAR(64),
    violation_type VARCHAR(100),
    violation_reason TEXT,
    raw_record JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_violation_type ON schema_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_detected_at ON schema_violations(detected_at);

CREATE TABLE IF NOT EXISTS deduplication_stats (
    stat_id SERIAL PRIMARY KEY,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    total_records BIGINT,
    duplicates_removed BIGINT,
    dedup_rate DOUBLE PRECISION,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomaly_scores (
    score_id SERIAL PRIMARY KEY,
    trip_id VARCHAR(64),
    anomaly_score DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    is_anomaly BOOLEAN,
    context_key VARCHAR(200),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trip_id ON anomaly_scores(trip_id);
CREATE INDEX IF NOT EXISTS idx_is_anomaly ON anomaly_scores(is_anomaly);

CREATE TABLE IF NOT EXISTS meta_metrics (
    metric_id SERIAL PRIMARY KEY,
    neighborhood VARCHAR(100),
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    volume BIGINT,
    null_rate DOUBLE PRECISION,
    violation_rate DOUBLE PRECISION,
    anomaly_rate DOUBLE PRECISION,
    avg_anomaly_score DOUBLE PRECISION,
    delta_score DOUBLE PRECISION,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_neighborhood ON meta_metrics(neighborhood);
CREATE INDEX IF NOT EXISTS idx_window_start ON meta_metrics(window_start);

CREATE TABLE IF NOT EXISTS drift_events (
    event_id SERIAL PRIMARY KEY,
    scenario VARCHAR(50),
    neighborhood VARCHAR(100),
    triggered_at TIMESTAMP,
    strategy VARCHAR(50),
    action_taken TEXT,
    recovery_time_sec INTEGER
);

CREATE INDEX IF NOT EXISTS idx_scenario ON drift_events(scenario);
CREATE INDEX IF NOT EXISTS idx_triggered_at ON drift_events(triggered_at);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    version VARCHAR(20),
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    artifact_uri TEXT,
    metrics JSONB
);

CREATE INDEX IF NOT EXISTS idx_deployed_at ON model_versions(deployed_at);
