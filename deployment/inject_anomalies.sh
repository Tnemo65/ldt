#!/bin/bash
# =============================================================================
# CA-DQStream - Anomaly Injection Script
# Usage:
#   docker cp c:/proj/ldt/deployment/inject_anomalies.sh ldt-kafka:/tmp/
#   docker exec ldt-kafka bash /tmp/inject_anomalies.sh
#
# Injects clearly-labeled anomaly records at ~2 records/sec for dashboard demo.
# =============================================================================

KAFKA="kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw"

echo "======================================================"
echo " CA-DQStream Anomaly Injection for Demo"
echo "======================================================"

NOW=$(date -u +"%Y-%m-%dT%H:%M:%S")
echo "Timestamp: $NOW"
echo ""

send() { printf '%s\n' "$1" | docker exec -i ldt-kafka $KAFKA 2>/dev/null; }
pause() { sleep 0.5; }

# =============================================================================
# BATCH 1: Normal valid records (baseline - should show as clean)
# =============================================================================
echo "--- Batch 1: Normal Valid Records ---"

send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:00:00","tpep_dropoff_datetime":"2026-05-13T14:15:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:00:10","tpep_dropoff_datetime":"2026-05-13T14:15:10","passenger_count":1,"trip_distance":2.5,"PULocationID":100,"DOLocationID":180,"fare_amount":9.50,"total_amount":12.50,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:00:20","tpep_dropoff_datetime":"2026-05-13T14:15:20","passenger_count":3,"trip_distance":4.0,"PULocationID":150,"DOLocationID":190,"fare_amount":14.00,"total_amount":18.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:00:30","tpep_dropoff_datetime":"2026-05-13T14:15:30","passenger_count":2,"trip_distance":5.5,"PULocationID":200,"DOLocationID":50,"fare_amount":18.50,"total_amount":23.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:00:40","tpep_dropoff_datetime":"2026-05-13T14:15:40","passenger_count":1,"trip_distance":1.8,"PULocationID":79,"DOLocationID":79,"fare_amount":7.50,"total_amount":10.00,"payment_type":1}'
pause; pause
echo "  Normal: 5 sent"

# =============================================================================
# BATCH 2: LAYER 1 - Schema Violations (missing required fields)
# =============================================================================
echo ""
echo "--- Batch 2: L1 Schema Violations ---"

# Missing trip_distance
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:01:00","tpep_dropoff_datetime":"2026-05-13T14:16:00","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:01:10","tpep_dropoff_datetime":"2026-05-13T14:16:10","passenger_count":2,"PULocationID":100,"DOLocationID":200,"fare_amount":15.00,"total_amount":18.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:01:20","tpep_dropoff_datetime":"2026-05-13T14:16:20","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":8.00,"total_amount":11.00,"payment_type":1}'
pause; pause
echo "  L1: Missing trip_distance: 3 sent"

# Missing fare_amount
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:02:00","tpep_dropoff_datetime":"2026-05-13T14:17:00","passenger_count":1,"trip_distance":2.0,"PULocationID":79,"DOLocationID":170,"total_amount":10.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:02:10","tpep_dropoff_datetime":"2026-05-13T14:17:10","passenger_count":3,"trip_distance":4.5,"PULocationID":150,"DOLocationID":200,"total_amount":20.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:02:20","tpep_dropoff_datetime":"2026-05-13T14:17:20","passenger_count":2,"trip_distance":3.0,"PULocationID":100,"DOLocationID":180,"total_amount":15.00,"payment_type":2}'
pause; pause
echo "  L1: Missing fare_amount: 3 sent"

# Invalid zone (>263 NYC)
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:03:00","tpep_dropoff_datetime":"2026-05-13T14:18:00","passenger_count":2,"trip_distance":5.0,"fare_amount":18.00,"PULocationID":999,"DOLocationID":500,"total_amount":22.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:03:10","tpep_dropoff_datetime":"2026-05-13T14:18:10","passenger_count":1,"trip_distance":3.5,"fare_amount":12.00,"PULocationID":0,"DOLocationID":999,"total_amount":16.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:03:20","tpep_dropoff_datetime":"2026-05-13T14:18:20","passenger_count":2,"trip_distance":6.0,"fare_amount":20.00,"PULocationID":300,"DOLocationID":0,"total_amount":25.00,"payment_type":1}'
pause; pause
echo "  L1: Invalid zone ID (>263): 3 sent"

# =============================================================================
# BATCH 3: LAYER 2 - Canary Rule Violations
# =============================================================================
echo ""
echo "--- Batch 3: L2 Canary Rule Violations ---"

# Rule 1: Negative fare
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:04:00","tpep_dropoff_datetime":"2026-05-13T14:19:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:04:10","tpep_dropoff_datetime":"2026-05-13T14:19:10","passenger_count":2,"trip_distance":3.0,"PULocationID":100,"DOLocationID":200,"fare_amount":-15.50,"total_amount":5.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:04:20","tpep_dropoff_datetime":"2026-05-13T14:19:20","passenger_count":1,"trip_distance":8.0,"PULocationID":150,"DOLocationID":79,"fare_amount":-99.00,"total_amount":0.00,"payment_type":1}'
pause; pause
echo "  L2-Rule1: Negative fare: 3 sent"

# Rule 2: Zero distance with fare
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:05:00","tpep_dropoff_datetime":"2026-05-13T14:06:00","passenger_count":1,"trip_distance":0.0,"PULocationID":79,"DOLocationID":79,"fare_amount":25.00,"total_amount":28.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:05:10","tpep_dropoff_datetime":"2026-05-13T14:06:10","passenger_count":2,"trip_distance":0.0,"PULocationID":100,"DOLocationID":100,"fare_amount":50.00,"total_amount":55.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:05:20","tpep_dropoff_datetime":"2026-05-13T14:06:20","passenger_count":1,"trip_distance":0.0,"PULocationID":150,"DOLocationID":150,"fare_amount":100.00,"total_amount":105.00,"payment_type":2}'
pause; pause
echo "  L2-Rule2: Zero dist + fare: 3 sent"

# Rule 3: passengers=0
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:06:00","tpep_dropoff_datetime":"2026-05-13T14:21:00","passenger_count":0,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:06:10","tpep_dropoff_datetime":"2026-05-13T14:21:10","passenger_count":0,"trip_distance":5.5,"PULocationID":100,"DOLocationID":200,"fare_amount":18.00,"total_amount":22.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:06:20","tpep_dropoff_datetime":"2026-05-13T14:21:20","passenger_count":0,"trip_distance":2.0,"PULocationID":150,"DOLocationID":79,"fare_amount":7.50,"total_amount":10.00,"payment_type":1}'
pause; pause
echo "  L2-Rule3: passengers=0: 3 sent"

# Rule 3: passengers>6
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:07:00","tpep_dropoff_datetime":"2026-05-13T14:22:00","passenger_count":9,"trip_distance":8.0,"PULocationID":79,"DOLocationID":170,"fare_amount":30.00,"total_amount":35.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:07:10","tpep_dropoff_datetime":"2026-05-13T14:22:10","passenger_count":20,"trip_distance":12.0,"PULocationID":100,"DOLocationID":200,"fare_amount":45.00,"total_amount":50.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:07:20","tpep_dropoff_datetime":"2026-05-13T14:22:20","passenger_count":7,"trip_distance":6.0,"PULocationID":150,"DOLocationID":79,"fare_amount":22.00,"total_amount":27.00,"payment_type":2}'
pause; pause
echo "  L2-Rule3: passengers>6: 3 sent"

# Rule 4: invalid payment (not 1-6)
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:08:00","tpep_dropoff_datetime":"2026-05-13T14:23:00","passenger_count":2,"trip_distance":4.0,"PULocationID":79,"DOLocationID":170,"fare_amount":15.00,"total_amount":18.00,"payment_type":99}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:08:10","tpep_dropoff_datetime":"2026-05-13T14:23:10","passenger_count":1,"trip_distance":3.0,"PULocationID":100,"DOLocationID":200,"fare_amount":10.00,"total_amount":13.00,"payment_type":0}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:08:20","tpep_dropoff_datetime":"2026-05-13T14:23:20","passenger_count":3,"trip_distance":5.5,"PULocationID":150,"DOLocationID":79,"fare_amount":20.00,"total_amount":25.00,"payment_type":77}'
pause; pause
echo "  L2-Rule4: invalid payment_type: 3 sent"

# Rule 5: extreme fare > $1000
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:09:00","tpep_dropoff_datetime":"2026-05-13T14:29:00","passenger_count":2,"trip_distance":50.0,"PULocationID":79,"DOLocationID":170,"fare_amount":2500.00,"total_amount":2550.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:09:10","tpep_dropoff_datetime":"2026-05-13T14:29:10","passenger_count":1,"trip_distance":80.0,"PULocationID":100,"DOLocationID":200,"fare_amount":5000.00,"total_amount":5050.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:09:20","tpep_dropoff_datetime":"2026-05-13T14:29:20","passenger_count":3,"trip_distance":100.0,"PULocationID":150,"DOLocationID":79,"fare_amount":10000.00,"total_amount":10050.00,"payment_type":1}'
pause; pause
echo "  L2-Rule5: extreme_fare >$1000: 3 sent"

# Rule 7: total < fare
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:10:00","tpep_dropoff_datetime":"2026-05-13T14:25:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":20.00,"total_amount":15.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:10:10","tpep_dropoff_datetime":"2026-05-13T14:25:10","passenger_count":2,"trip_distance":8.0,"PULocationID":100,"DOLocationID":200,"fare_amount":35.00,"total_amount":25.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:10:20","tpep_dropoff_datetime":"2026-05-13T14:25:20","passenger_count":3,"trip_distance":3.0,"PULocationID":150,"DOLocationID":79,"fare_amount":50.00,"total_amount":30.00,"payment_type":1}'
pause; pause
echo "  L2-Rule7: total<fare: 3 sent"

# Multiple violations at once
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:11:00","tpep_dropoff_datetime":"2026-05-13T14:26:00","passenger_count":0,"trip_distance":0.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-10.00,"total_amount":5.00,"payment_type":99}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:11:10","tpep_dropoff_datetime":"2026-05-13T14:26:10","passenger_count":9,"trip_distance":0.0,"PULocationID":300,"DOLocationID":999,"fare_amount":-50.00,"total_amount":0.00,"payment_type":0}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:11:20","tpep_dropoff_datetime":"2026-05-13T14:26:20","passenger_count":0,"trip_distance":0.0,"PULocationID":999,"DOLocationID":999,"fare_amount":-999.00,"total_amount":-500.00,"payment_type":77}'
pause; pause
echo "  L2-Multi: 3 sent (each has 4 violations)"

# =============================================================================
# BATCH 4: More normal records
# =============================================================================
echo ""
echo "--- Batch 4: Normal Records (post-violations) ---"

send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:12:00","tpep_dropoff_datetime":"2026-05-13T14:27:00","passenger_count":3,"trip_distance":6.5,"PULocationID":170,"DOLocationID":79,"fare_amount":22.00,"total_amount":27.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:12:10","tpep_dropoff_datetime":"2026-05-13T14:27:10","passenger_count":1,"trip_distance":2.0,"PULocationID":79,"DOLocationID":170,"fare_amount":8.50,"total_amount":11.50,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:12:20","tpep_dropoff_datetime":"2026-05-13T14:27:20","passenger_count":2,"trip_distance":4.5,"PULocationID":100,"DOLocationID":200,"fare_amount":16.00,"total_amount":20.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:12:30","tpep_dropoff_datetime":"2026-05-13T14:27:30","passenger_count":4,"trip_distance":7.0,"PULocationID":150,"DOLocationID":50,"fare_amount":24.00,"total_amount":29.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:12:40","tpep_dropoff_datetime":"2026-05-13T14:27:40","passenger_count":2,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":11.00,"total_amount":14.00,"payment_type":1}'
pause; pause
echo "  Normal: 5 sent"

# =============================================================================
# BATCH 5: Extreme ML + Canary anomalies
# =============================================================================
echo ""
echo "--- Batch 5: Extreme ML + Canary Anomalies ---"

send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:13:00","tpep_dropoff_datetime":"2026-05-13T14:28:00","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:13:10","tpep_dropoff_datetime":"2026-05-13T14:28:10","passenger_count":6,"trip_distance":88.8,"PULocationID":229,"DOLocationID":138,"fare_amount":888.88,"total_amount":920.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:13:20","tpep_dropoff_datetime":"2026-05-13T14:28:20","passenger_count":6,"trip_distance":77.7,"PULocationID":100,"DOLocationID":200,"fare_amount":777.77,"total_amount":800.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:13:30","tpep_dropoff_datetime":"2026-05-13T14:28:30","passenger_count":6,"trip_distance":66.6,"PULocationID":150,"DOLocationID":50,"fare_amount":666.66,"total_amount":700.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:13:40","tpep_dropoff_datetime":"2026-05-13T14:28:40","passenger_count":6,"trip_distance":55.5,"PULocationID":79,"DOLocationID":170,"fare_amount":555.55,"total_amount":580.00,"payment_type":1}'
pause; pause
echo "  Extreme fare+dist: 5 sent"

# Edge case: near-zero
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:14:00","tpep_dropoff_datetime":"2026-05-13T14:15:00","passenger_count":1,"trip_distance":0.0,"PULocationID":1,"DOLocationID":1,"fare_amount":0.01,"total_amount":0.01,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:14:10","tpep_dropoff_datetime":"2026-05-13T14:15:10","passenger_count":1,"trip_distance":0.0,"PULocationID":1,"DOLocationID":1,"fare_amount":0.01,"total_amount":0.01,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:14:20","tpep_dropoff_datetime":"2026-05-13T14:15:20","passenger_count":1,"trip_distance":0.0,"PULocationID":1,"DOLocationID":1,"fare_amount":0.01,"total_amount":0.01,"payment_type":1}'
pause; pause
echo "  Edge case (near-zero): 3 sent"

# =============================================================================
# BATCH 6: Final normal burst
# =============================================================================
echo ""
echo "--- Batch 6: Normal Burst ---"

send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:15:00","tpep_dropoff_datetime":"2026-05-13T14:30:00","passenger_count":2,"trip_distance":8.0,"PULocationID":100,"DOLocationID":200,"fare_amount":28.00,"total_amount":33.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:15:10","tpep_dropoff_datetime":"2026-05-13T14:30:10","passenger_count":3,"trip_distance":10.0,"PULocationID":79,"DOLocationID":170,"fare_amount":35.00,"total_amount":40.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:15:20","tpep_dropoff_datetime":"2026-05-13T14:30:20","passenger_count":1,"trip_distance":4.5,"PULocationID":100,"DOLocationID":79,"fare_amount":16.00,"total_amount":20.00,"payment_type":2}'
pause; pause
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-13T14:15:30","tpep_dropoff_datetime":"2026-05-13T14:30:30","passenger_count":2,"trip_distance":6.0,"PULocationID":150,"DOLocationID":200,"fare_amount":22.00,"total_amount":27.00,"payment_type":1}'
pause; pause
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-13T14:15:40","tpep_dropoff_datetime":"2026-05-13T14:30:40","passenger_count":4,"trip_distance":9.0,"PULocationID":79,"DOLocationID":50,"fare_amount":32.00,"total_amount":37.00,"payment_type":2}'
pause; pause
echo "  Normal burst: 5 sent"

echo ""
echo "======================================================"
echo " INJECTION COMPLETE"
echo "======================================================"
echo "Summary:"
echo "  L1 Schema violations: 9 records"
echo "  L2 Canary violations: 27 records"
echo "  Extreme anomalies: 5 records"
echo "  Edge case: 3 records"
echo "  Normal records: 20 records"
echo "  TOTAL: ~64 records at ~2/sec (~32 seconds)"
echo ""
echo "Dashboard timeline:"
echo "  0-15s:   Normal baseline (5 records)"
echo "  15-30s:  L1 Schema violations (9 records)"
echo "  30-90s:  L2 Canary violations (27 records)"
echo "  90-105s: Extreme anomalies (5 records)"
echo "  105-120s:Normal burst (20 records)"
echo ""
echo "Expected dashboard panels:"
echo "  - Schema Violations (5m): spike at ~14:01-14:03"
echo "  - Canary Violation Rate: spike at ~14:04-14:11"
echo "  - ML Anomaly Rate: spike at ~14:13-14:14"
echo "  - IEC Drift: if enough anomalies in 1-min window"
echo "======================================================"
