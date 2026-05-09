#!/usr/bin/env python3
import psycopg2
conn = psycopg2.connect(host='postgres', port=5432, dbname='dq_pipeline', user='cadqstream', password='cadqstream123')
cur = conn.cursor()
cur.execute('SELECT 1')
print('Connected! Result:', cur.fetchone())
cur.execute("INSERT INTO anomaly_scores (trip_id, anomaly_score, threshold, is_anomaly) VALUES ('test123', 0.5, 0.5, false)")
conn.commit()
cur.execute('SELECT COUNT(*) FROM anomaly_scores')
print('Count:', cur.fetchone())
cur.close()
conn.close()
print('SUCCESS')
