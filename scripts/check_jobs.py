#!/usr/bin/env python3
import json, urllib.request

try:
    resp = urllib.request.urlopen("http://localhost:8081/jobs", timeout=5)
    d = json.loads(resp.read())
    jobs = d.get("jobs", [])
    print(f"Jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j['id'][:20]} - {j['status']}")
except Exception as e:
    print(f"Error: {e}")
