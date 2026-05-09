#!/usr/bin/env python3
import psycopg2
try:
    conn = psycopg2.connect(host='postgres', port=5432, dbname='dq_pipeline', user='cadqstream', password='cadqstream123')
    print('Connected!')
    cur = conn.cursor()
    cur.execute('SELECT 1')
    print('Query:', cur.fetchone())
    conn.close()
except Exception as e:
    print('ERROR:', e)
