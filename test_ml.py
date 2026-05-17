#!/usr/bin/env python3
import requests, json

data = {
    "features": [[
        900.0, 3.5, 15.5, 2.5, 0.33, 0.95, 0.14, 0.0, 2.0, 100.0,
        170.0, 5.0, 1.3, 0.16, 0.1, 0.05, 1.0, 1.0, 0.0, 1.0,
        0.87, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.1, 0.9,
        0.15, 0.85, 0.25, 0.75
    ]]
}

r = requests.post("http://localhost:8000/predict", json=data, timeout=15)
print(f"Status: {r.status_code}")
print(f"Response: {r.text}")
