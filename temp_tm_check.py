import json, urllib.request
try:
    r = urllib.request.urlopen("http://localhost:8081/taskmanagers", timeout=5)
    data = json.loads(r.read())
    print("TaskManagers:", len(data.get("taskmanagers", [])))
    for tm in data.get("taskmanagers", []):
        print(f"  {tm.get('id')}: slots={tm.get('slotsNumber')}, free={tm.get('freeSlots')}, status={tm.get('status')}")
except Exception as e:
    print(f"Error: {e}")

# Also check the TM logs
print("\nChecking TM logs:")
import pathlib
log_files = list(pathlib.Path("/opt/flink/log").glob("flink--taskexecutor*.log"))
for lf in sorted(log_files)[-3:]:
    print(f"\n{lf.name}:")
    lines = lf.read_text().splitlines()
    for line in lines[-5:]:
        print(f"  {line}")
