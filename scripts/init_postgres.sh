#!/bin/bash
# scripts/init_postgres.sh

CONTAINER="ldt-postgres-1"
DB="dq_pipeline"
USER="cadqstream"

echo "Initializing PostgreSQL schema..."

docker exec -i $CONTAINER psql -U $USER -d $DB < sql/schema.sql

if [ $? -eq 0 ]; then
  echo "✅ Schema initialized"

  echo ""
  echo "Tables:"
  docker exec $CONTAINER psql -U $USER -d $DB -c "\dt"
else
  echo "❌ Schema initialization failed"
  exit 1
fi
