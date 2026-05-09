import json, urllib.request, pathlib
r = urllib.request.urlopen("http://localhost:8081/overview", timeout=5)
data = json.loads(r.read())
print("TMs=" + str(data.get("taskmanagers")) + ", Slots=" + str(data.get("slots-total")) + "/" + str(data.get("slots-available")))
log_dir = pathlib.Path("/opt/flink/log")
files = list(log_dir.glob("flink--taskexecutor-*-*.log"))
files.sort(key=lambda x: x.stat().st_mtime)
if files:
    lf = files[-1]
    lines = lf.read_text().splitlines()
    recent = [l for l in lines if "14:" in l or "15:" in l]
    print("Latest TM log: " + lf.name + ", recent lines: " + str(len(recent)))
    for line in recent[-3:]:
        print("  " + line)
