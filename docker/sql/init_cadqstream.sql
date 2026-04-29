-- CA-DQStream schema (Layer 1-4 + quality metrics)
CREATE TABLE IF NOT EXISTS taxi_clean (
    id              BIGSERIAL PRIMARY KEY,
    VendorID        INTEGER,
    tpep_pickup     TIMESTAMP NOT NULL,
    tpep_dropoff    TIMESTAMP NOT NULL,
    passenger_count REAL,
    trip_distance   REAL,
    RatecodeID      INTEGER,
    PULocationID    INTEGER,
    DOLocationID    INTEGER,
    payment_type    INTEGER,
    fare_amount     REAL,
    extra           REAL,
    mta_tax         REAL,
    tip_amount      REAL,
    tolls_amount    REAL,
    improvement_surcharge REAL,
    total_amount    REAL,
    congestion_surcharge REAL,
    Airport_fee     REAL,
    duration_min    REAL,
    speed_mph       REAL,
    tip_ratio       REAL,
    time_slot       INTEGER,
    ratecode_type   VARCHAR(20),
    if_score        REAL,
    lof_score       REAL,
    svm_score       REAL,
    ae_score        REAL,
    ensemble_score  REAL,
    priority        VARCHAR(10),
    batch_id        BIGINT,
    processed_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_taxi_clean_pickup ON taxi_clean(tpep_pickup);
CREATE INDEX IF NOT EXISTS idx_taxi_clean_priority ON taxi_clean(priority);

CREATE TABLE IF NOT EXISTS schema_violations (
    id              BIGSERIAL PRIMARY KEY,
    raw_data        JSONB NOT NULL,
    violation_type  VARCHAR(50) NOT NULL,
    rejected_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rule_violations (
    id              BIGSERIAL PRIMARY KEY,
    taxi_clean_id   BIGINT REFERENCES taxi_clean(id),
    layer           INTEGER NOT NULL,
    rule_name       VARCHAR(100) NOT NULL,
    feature_name    VARCHAR(50),
    feature_value   REAL,
    z_score         REAL,
    context_group   VARCHAR(50),
    priority        VARCHAR(10),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rule_violations_layer ON rule_violations(layer);
CREATE INDEX IF NOT EXISTS idx_rule_violations_context ON rule_violations(context_group);

CREATE TABLE IF NOT EXISTS ml_anomalies (
    id              BIGSERIAL PRIMARY KEY,
    taxi_clean_id   BIGINT REFERENCES taxi_clean(id),
    if_anomaly      BOOLEAN DEFAULT FALSE,
    lof_anomaly     BOOLEAN DEFAULT FALSE,
    svm_anomaly     BOOLEAN DEFAULT FALSE,
    ae_anomaly      BOOLEAN DEFAULT FALSE,
    ensemble_anomaly BOOLEAN DEFAULT FALSE,
    if_score        REAL,
    lof_score       REAL,
    svm_score       REAL,
    ae_score        REAL,
    ensemble_score  REAL,
    anomaly_type    VARCHAR(50),
    confidence      VARCHAR(10),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_anomalies_time ON ml_anomalies(created_at);
CREATE INDEX IF NOT EXISTS idx_ml_anomalies_type ON ml_anomalies(anomaly_type);

CREATE TABLE IF NOT EXISTS quality_metrics (
    id              BIGSERIAL PRIMARY KEY,
    metric_ts       TIMESTAMP NOT NULL,
    context_group   VARCHAR(50),
    null_rate       REAL,
    violation_rate_l1 REAL,
    violation_rate_l2 REAL,
    violation_rate_l3 REAL,
    violation_rate_l4 REAL,
    anomaly_rate    REAL,
    record_count    BIGINT,
    volume_ratio    REAL,
    consumer_lag_ms BIGINT,
    dedup_rate      REAL,
    drift_detected  BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_ts ON quality_metrics(metric_ts);

CREATE TABLE IF NOT EXISTS ml_exp_results (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   VARCHAR(50) NOT NULL,
    approach        VARCHAR(50) NOT NULL,
    threshold       REAL,
    precision       REAL,
    recall          REAL,
    fpr             REAL,
    f1              REAL,
    accuracy        REAL,
    weights         JSONB,
    trained_at      TIMESTAMP DEFAULT NOW()
);

SELECT 'CA-DQStream tables created successfully' AS status;
