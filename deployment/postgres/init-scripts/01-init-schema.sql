-- =============================================================================
-- CA-DQStream PostgreSQL Initialization Script
-- Schema for all 4 pipeline layers: Baseline -> Dual-Branch -> Rendezvous+Meta -> IEC
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_prewarm";

-- =============================================================================
-- LAYER 1: Raw Data Table (Baseline Validation output)
-- =============================================================================

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
    trip_id_hash VARCHAR(32),
    neighborhood VARCHAR(50),
    layer1_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE taxi_trips_raw IS 'Layer 1 output: Deduplicated, schema-validated taxi records';
COMMENT ON COLUMN taxi_trips_raw.trip_id_hash IS 'MurmurHash3 of trip_id for fast lookups';
COMMENT ON COLUMN taxi_trips_raw.neighborhood IS 'Spatial grouping: manhattan, brooklyn, queens, bronx, airport, staten_island';

CREATE INDEX IF NOT EXISTS idx_trips_pickup_dt ON taxi_trips_raw(pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_trips_dropoff_dt ON taxi_trips_raw(dropoff_datetime);
CREATE INDEX IF NOT EXISTS idx_trips_pickup_loc ON taxi_trips_raw(pickup_location_id);
CREATE INDEX IF NOT EXISTS idx_trips_dropoff_loc ON taxi_trips_raw(dropoff_location_id);
CREATE INDEX IF NOT EXISTS idx_trips_neighborhood ON taxi_trips_raw(neighborhood);
CREATE INDEX IF NOT EXISTS idx_trips_ingestion ON taxi_trips_raw(ingestion_timestamp);
CREATE INDEX IF NOT EXISTS idx_trips_vendor ON taxi_trips_raw(vendor_id) WHERE vendor_id IS NOT NULL;

-- =============================================================================
-- LAYER 1: Schema Violations Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_violations (
    violation_id BIGSERIAL PRIMARY KEY,
    trip_id VARCHAR(64),
    violation_type VARCHAR(100) NOT NULL,
    violation_reason TEXT,
    raw_record JSONB,
    layer1_layer VARCHAR(10) DEFAULT 'L1',
    field_name VARCHAR(100),
    field_value TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    kafka_offset BIGINT,
    kafka_partition INTEGER
);

COMMENT ON TABLE schema_violations IS 'Layer 1 output: Records failing schema validation';

CREATE INDEX IF NOT EXISTS idx_violations_trip_id ON schema_violations(trip_id);
CREATE INDEX IF NOT EXISTS idx_violations_type ON schema_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_violations_detected ON schema_violations(detected_at);
CREATE INDEX IF NOT EXISTS idx_violations_raw ON schema_violations USING GIN(raw_record);

-- =============================================================================
-- LAYER 2: Canary (Rule-Based) Violations
-- =============================================================================

CREATE TABLE IF NOT EXISTS canary_violations (
    violation_id BIGSERIAL PRIMARY KEY,
    trip_id VARCHAR(64) NOT NULL,
    violation_types TEXT[],  -- Array of rule names: negative_fare, zero_distance_with_fare, etc.
    violation_count INTEGER DEFAULT 0,
    fare_amount DOUBLE PRECISION,
    trip_distance DOUBLE PRECISION,
    passenger_count INTEGER,
    payment_type INTEGER,
    pickup_datetime TIMESTAMP,
    final_decision VARCHAR(20),  -- ANOMALY or CLEAN
    decision_source VARCHAR(20) DEFAULT 'canary_rule',
    confidence DOUBLE PRECISION,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE canary_violations IS 'Layer 2 output: Hard rule violations (Canary branch)';

CREATE INDEX IF NOT EXISTS idx_canary_trip_id ON canary_violations(trip_id);
CREATE INDEX IF NOT EXISTS idx_canary_types ON canary_violations USING GIN(violation_types);
CREATE INDEX IF NOT EXISTS idx_canary_decision ON canary_violations(final_decision);
CREATE INDEX IF NOT EXISTS idx_canary_detected ON canary_violations(detected_at);
CREATE INDEX IF NOT EXISTS idx_canary_fare ON canary_violations(fare_amount) WHERE fare_amount IS NOT NULL;

-- =============================================================================
-- LAYER 2: Anomaly Scores (Complex/ML branch)
-- =============================================================================

CREATE TABLE IF NOT EXISTS anomaly_scores (
    score_id BIGSERIAL PRIMARY KEY,
    trip_id VARCHAR(64) NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    context_key VARCHAR(200),
    neighborhood VARCHAR(50),
    model_version VARCHAR(50),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE anomaly_scores IS 'Layer 2 output: ML anomaly detection scores';

CREATE INDEX IF NOT EXISTS idx_scores_trip_id ON anomaly_scores(trip_id);
CREATE INDEX IF NOT EXISTS idx_scores_anomaly ON anomaly_scores(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_scores_neighborhood ON anomaly_scores(neighborhood);
CREATE INDEX IF NOT EXISTS idx_scores_scored_at ON anomaly_scores(scored_at);
CREATE INDEX IF NOT EXISTS idx_scores_score ON anomaly_scores(anomaly_score) WHERE is_anomaly = TRUE;

-- =============================================================================
-- LAYER 3: Meta-Metrics (Windowed Aggregates)
-- =============================================================================

CREATE TABLE IF NOT EXISTS meta_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    neighborhood VARCHAR(100) NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    volume BIGINT NOT NULL,
    null_rate DOUBLE PRECISION,
    violation_rate DOUBLE PRECISION,
    anomaly_rate DOUBLE PRECISION,
    avg_anomaly_score DOUBLE PRECISION,
    delta_score DOUBLE PRECISION,
    total_records BIGINT,
    clean_records BIGINT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(neighborhood, window_start)
);

COMMENT ON TABLE meta_metrics IS 'Layer 3 output: 1-minute windowed meta-metrics per neighborhood';
COMMENT ON COLUMN meta_metrics.delta_score IS 'Change in anomaly_rate from previous window';

CREATE INDEX IF NOT EXISTS idx_meta_neighborhood ON meta_metrics(neighborhood);
CREATE INDEX IF NOT EXISTS idx_meta_window_start ON meta_metrics(window_start);
CREATE INDEX IF NOT EXISTS idx_meta_window_end ON meta_metrics(window_end);
CREATE INDEX IF NOT EXISTS idx_meta_recorded ON meta_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_meta_anomaly_rate ON meta_metrics(anomaly_rate) WHERE anomaly_rate > 0.1;

-- =============================================================================
-- LAYER 4: Drift Events (IEC Decisions)
-- =============================================================================

CREATE TABLE IF NOT EXISTS drift_events (
    event_id BIGSERIAL PRIMARY KEY,
    scenario VARCHAR(50) NOT NULL,
    neighborhood VARCHAR(100),
    metric_name VARCHAR(100),
    drift_indicator VARCHAR(50),
    drift_magnitude DOUBLE PRECISION,
    neighborhood_count INTEGER,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(50) NOT NULL,
    iec_confidence DOUBLE PRECISION,
    action_taken TEXT,
    recovery_time_sec INTEGER,
    window_start TIMESTAMP,
    window_end TIMESTAMP
);

COMMENT ON TABLE drift_events IS 'Layer 4 output: IEC drift detection and strategy decisions';

CREATE INDEX IF NOT EXISTS idx_drift_scenario ON drift_events(scenario);
CREATE INDEX IF NOT EXISTS idx_drift_neighborhood ON drift_events(neighborhood);
CREATE INDEX IF NOT EXISTS idx_drift_strategy ON drift_events(strategy);
CREATE INDEX IF NOT EXISTS idx_drift_triggered ON drift_events(triggered_at);
CREATE INDEX IF NOT EXISTS idx_drift_magnitude ON drift_events(drift_magnitude) WHERE drift_magnitude > 0.1;

-- =============================================================================
-- IEC: Model Versions
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_versions (
    version_id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    artifact_uri TEXT,
    mlflow_run_id VARCHAR(50),
    training_start TIMESTAMP,
    training_end TIMESTAMP,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    metrics JSONB,
    hyperparameters JSONB,
    notes TEXT
);

COMMENT ON TABLE model_versions IS 'IEC: Model version tracking with MLflow integration';

CREATE INDEX IF NOT EXISTS idx_models_name ON model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_models_active ON model_versions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_models_deployed ON model_versions(deployed_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_mlflow ON model_versions(mlflow_run_id) WHERE mlflow_run_id IS NOT NULL;

-- =============================================================================
-- Pipeline: Checkpoint Metadata
-- =============================================================================

CREATE TABLE IF NOT EXISTS checkpoint_metadata (
    checkpoint_id BIGSERIAL PRIMARY KEY,
    job_name VARCHAR(200) NOT NULL,
    job_id VARCHAR(100),
    checkpoint_path TEXT NOT NULL,
    external_path TEXT,
    state_size BIGINT,
    end_to_end_duration_ms BIGINT,
    alignment_buffered BIGINT,
    processed_records BIGINT,
    persisted_records BIGINT,
    num_subtasks INTEGER,
    num_acknowledged_subtasks INTEGER,
    trigger_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_savepoint BOOLEAN DEFAULT FALSE,
    is_completed BOOLEAN DEFAULT FALSE
);

COMMENT ON TABLE checkpoint_metadata IS 'Flink checkpoint tracking for recovery and auditing';

CREATE INDEX IF NOT EXISTS idx_ckpt_job ON checkpoint_metadata(job_name);
CREATE INDEX IF NOT EXISTS idx_ckpt_triggered ON checkpoint_metadata(trigger_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ckpt_size ON checkpoint_metadata(state_size);

-- =============================================================================
-- Pipeline: Processing Statistics
-- =============================================================================

CREATE TABLE IF NOT EXISTS pipeline_stats (
    stat_id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metric_labels JSONB,
    source_service VARCHAR(50),
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pstats_metric ON pipeline_stats(metric_name);
CREATE INDEX IF NOT EXISTS idx_pstats_labels ON pipeline_stats USING GIN(metric_labels);
CREATE INDEX IF NOT EXISTS idx_pstats_recorded ON pipeline_stats(recorded_at DESC);

-- =============================================================================
-- Analytics Views
-- =============================================================================

CREATE OR REPLACE VIEW v_anomaly_trends AS
SELECT
    date_trunc('minute', m.window_start) AS time_bucket,
    m.neighborhood,
    AVG(m.anomaly_rate) AS avg_anomaly_rate,
    AVG(m.violation_rate) AS avg_violation_rate,
    SUM(m.volume) AS total_volume,
    COUNT(*) AS window_count
FROM meta_metrics m
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

CREATE OR REPLACE VIEW v_drift_summary AS
SELECT
    d.neighborhood,
    d.strategy,
    COUNT(*) AS event_count,
    AVG(d.drift_magnitude) AS avg_magnitude,
    MIN(d.triggered_at) AS first_event,
    MAX(d.triggered_at) AS last_event,
    AVG(d.iec_confidence) AS avg_confidence
FROM drift_events d
GROUP BY 1, 2
ORDER BY event_count DESC;

CREATE OR REPLACE VIEW v_layer1_quality AS
SELECT
    date_trunc('hour', recorded_at) AS time_bucket,
    violation_type,
    COUNT(*) AS violation_count,
    COUNT(*) FILTER (WHERE trip_id IS NOT NULL) AS with_trip_id
FROM schema_violations
GROUP BY 1, 2
ORDER BY time_bucket DESC;
