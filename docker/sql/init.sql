-- =====================================================
-- CA-DQStream PostgreSQL Schema
-- =====================================================

-- Clean taxi records (Layer 0 output)
CREATE TABLE IF NOT EXISTS taxi_clean (
    id              BIGSERIAL PRIMARY KEY,
    VendorID        INTEGER,
    tpep_pickup     TIMESTAMP NOT NULL,
    tpep_dropoff    TIMESTAMP NOT NULL,
    passenger_count REAL,
    trip_distance   REAL,
    RatecodeID      INTEGER,
    store_and_fwd_flag VARCHAR(1),
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
    -- Derived features (for ML)
    duration_min    REAL,
    speed_mph      REAL,
    tip_ratio      REAL,
    -- Context
    time_slot       INTEGER,
    ratecode_type   VARCHAR(20),
    -- ML scores
    if_score        REAL,
    lof_score       REAL,
    svm_score       REAL,
    ae_score        REAL,
    ensemble_score  REAL,
    priority        VARCHAR(10),
    -- Metadata
    batch_id        BIGINT,
    processed_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_taxi_clean_pickup ON taxi_clean(tpep_pickup);
CREATE INDEX IF NOT EXISTS idx_taxi_clean_location ON taxi_clean(PULocationID);
CREATE INDEX IF NOT EXISTS idx_taxi_clean_priority ON taxi_clean(priority);
CREATE INDEX IF NOT EXISTS idx_taxi_clean_batch ON taxi_clean(batch_id);

-- Schema violations (Layer 1 output)
CREATE TABLE IF NOT EXISTS schema_violations (
    id              BIGSERIAL PRIMARY KEY,
    raw_data        JSONB NOT NULL,
    violation_type  VARCHAR(50) NOT NULL,
    violation_msg   TEXT,
    rejected_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schema_violations_type ON schema_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_schema_violations_time ON schema_violations(rejected_at);

-- Hard rule violations (Layer 2 output)
CREATE TABLE IF NOT EXISTS rule_violations (
    id              BIGSERIAL PRIMARY KEY,
    taxi_clean_id   BIGINT REFERENCES taxi_clean(id),
    layer           INTEGER NOT NULL,          -- 2=hard rules, 3=context-aware
    rule_name       VARCHAR(100) NOT NULL,
    feature_name    VARCHAR(50),
    feature_value   REAL,
    threshold_low   REAL,
    threshold_high  REAL,
    z_score         REAL,
    context_group   VARCHAR(50),
    priority        VARCHAR(10),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rule_violations_layer ON rule_violations(layer);
CREATE INDEX IF NOT EXISTS idx_rule_violations_rule ON rule_violations(rule_name);
CREATE INDEX IF NOT EXISTS idx_rule_violations_context ON rule_violations(context_group);
CREATE INDEX IF NOT EXISTS idx_rule_violations_time ON rule_violations(created_at);

-- ML anomalies (Layer 4 output)
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
    anomaly_type    VARCHAR(50),       -- 'meter_tampering', 'short_trip_fraud', etc.
    confidence      VARCHAR(10),        -- 'HIGH', 'MEDIUM', 'LOW'
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_anomalies_time ON ml_anomalies(created_at);
CREATE INDEX IF NOT EXISTS idx_ml_anomalies_type ON ml_anomalies(anomaly_type);

-- Quality metrics (ADWIN meta-level)
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
    volume_ratio    REAL,              -- current volume / avg volume
    consumer_lag_ms BIGINT,
    dedup_rate      REAL,
    drift_detected  BOOLEAN DEFAULT FALSE,
    drift_type      VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_ts ON quality_metrics(metric_ts);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_context ON quality_metrics(context_group);

-- Baseline statistics (updated by Flink periodically)
CREATE TABLE IF NOT EXISTS baseline_stats (
    id              BIGSERIAL PRIMARY KEY,
    context_group   VARCHAR(50) UNIQUE NOT NULL,
    feature_name    VARCHAR(50) NOT NULL,
    count           BIGINT NOT NULL,
    mean            REAL NOT NULL,
    std             REAL NOT NULL,
    min_val         REAL,
    max_val         REAL,
    p5              REAL,
    p50             REAL,
    p95             REAL,
    window_start    TIMESTAMP NOT NULL,
    window_end      TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_baseline_context ON baseline_stats(context_group);
CREATE INDEX IF NOT EXISTS idx_baseline_feature ON baseline_stats(feature_name);

-- ML experiment results (for model comparison)
CREATE TABLE IF NOT EXISTS ml_exp_results (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   VARCHAR(50) NOT NULL,
    dataset         VARCHAR(50) NOT NULL,     -- 'train_2024', 'test_2025', 'synthetic'
    approach        VARCHAR(50) NOT NULL,      -- 'IF', 'LOF', 'SVM', 'AE', 'LOF+SVM(tuned)', etc.
    threshold       REAL,
    precision       REAL,
    recall          REAL,
    fpr             REAL,
    f1              REAL,
    accuracy        REAL,
    tp              BIGINT,
    fp              BIGINT,
    fn              BIGINT,
    tn              BIGINT,
    weights         JSONB,
    trained_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_exp_approach ON ml_exp_results(approach);
CREATE INDEX IF NOT EXISTS idx_ml_exp_dataset ON ml_exp_results(dataset);
CREATE INDEX IF NOT EXISTS idx_ml_exp_f1 ON ml_exp_results(f1 DESC);

-- Kafka consumer offsets (for monitoring)
CREATE TABLE IF NOT EXISTS kafka_offsets (
    id              BIGSERIAL PRIMARY KEY,
    topic           VARCHAR(100) NOT NULL,
    partition       INTEGER NOT NULL,
    consumer_group  VARCHAR(100) NOT NULL,
    current_offset  BIGINT,
    log_end_offset  BIGINT,
    lag             BIGINT,
    recorded_at     TIMESTAMP DEFAULT NOW(),
    UNIQUE(topic, partition, consumer_group)
);

CREATE INDEX IF NOT EXISTS idx_kafka_offsets_topic ON kafka_offsets(topic);
