import json, urllib.request, pathlib

# Check current TM status
try:
    r = urllib.request.urlopen("http://localhost:8081/taskmanagers", timeout=5)
    data = json.loads(r.read())
    print("TaskManagers:", len(data.get("taskmanagers", [])))
    for tm in data.get("taskmanagers", []):
        print(f"  {tm.get('id')}: slots={tm.get('slotsNumber')}, free={tm.get('freeSlots')}, status={tm.get('status')}")
except Exception as e:
    print(f"Error: {e}")

# Check JM's view of slots
try:
    r = urllib.request.urlopen("http://localhost:8081/overview", timeout=5)
    data = json.loads(r.read())
    print(f"\nFlink Overview:")
    print(f"  TaskManagers: {data.get('taskmanagers')}")
    print(f"  Slots total: {data.get('slots-total')}")
    print(f"  Slots available: {data.get('slots-available')}")
except Exception as e:
    print(f"Overview error: {e}")

# Check latest TM logs
print("\nChecking latest TM log entries:")
log_files = sorted(pathlib.Path("/opt/flink/log").glob("flink--taskexecutor-*-*.log"))
latest = log_files[-3:]
for lf in latest:
    name = lf.name
    # Get the last 3 lines
    lines = lf.read_text().splitlines()
    last_lines = [l for l in lines if "14:0" in l or "13:5" in l][-3:]
    print(f"\n{name}:")
    for line in last_lines:
        print(f"  {line}")
