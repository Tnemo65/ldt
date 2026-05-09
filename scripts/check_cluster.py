#!/usr/bin/env python3
import json
import urllib.request

try:
    resp = urllib.request.urlopen("http://localhost:8081/taskmanagers", timeout=5)
    d = json.loads(resp.read())
    tms = d.get("taskmanagers", [])
    print(f"TaskManagers online: {len(tms)}")
    for tm in tms:
        print(f"  {tm.get('id','?')[:20]} - {tm.get('status','?')} - slots: {tm.get('slotsTotal','?')}")
except Exception as e:
    print(f"Error: {e}")
