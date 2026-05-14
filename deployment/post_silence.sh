#!/bin/sh
DATA='{"matchers":[{"name":"alertname","value":"PostgreSQLDown","isRegex":false},{"name":"job","value":"postgres-exporter","isRegex":false}],"startsAt":"2026-05-13T00:00:00Z","endsAt":"2027-01-01T00:00:00Z","createdBy":"admin","comment":"PostgreSQL_removed_false_positive"}
'
wget -qO- --post-data="$DATA" --header="Content-Type: application/json" http://localhost:9090/api/v1/silences