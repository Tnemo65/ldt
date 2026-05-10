-- =============================================================================
-- PostgreSQL Performance Tuning - pg_tune.conf
-- =============================================================================

-- Applied via postgres.conf overrides
-- These settings are tuned for a 4-8GB PostgreSQL instance in Docker

-- Memory settings
shared_buffers = 512MB
effective_cache_size = 1GB
work_mem = 32MB
maintenance_work_mem = 128MB
temp_buffers = 16MB

-- Write-ahead log
wal_buffers = 16MB
min_wal_size = 256MB
max_wal_size = 1GB
checkpoint_completion_target = 0.9

-- Query planner
random_page_cost = 1.1
effective_io_concurrency = 200
default_statistics_target = 100

-- Logging
log_destination = 'csvlog'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 256MB
log_min_duration_statement = 1000
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

-- Autovacuum tuning
autovacuum_max_workers = 3
autovacuum_naptime = 30s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.02

-- Connection pooling friendly
max_connections = 100
