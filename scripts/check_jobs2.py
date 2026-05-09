#!/usr/bin/env python3
import json, urllib.request

try:
    resp = urllib.request.urlopen("http://localhost:8081/jobs/overview", timeout=5)
    d = json.loads(resp.read())
    jobs = d if isinstance(d, list) else d.get("jobs", [])
    if isinstance(d, dict) and "jobs" in d:
        jobs = d["jobs"]
    print(f"Jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j.get('id','?')[:20]} - {j.get('status','?')} - name: {j.get('name','?')}")
except Exception as e:
    print(f"Error: {e}")
