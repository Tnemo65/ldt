import json, subprocess
result = subprocess.run(["docker", "inspect", "ldt-flink-jobmanager", "--format", "{{json .Mounts}}"], capture_output=True, text=True)
mounts = json.loads(result.stdout)
for m in mounts:
    print(f"  {m['Source']} -> {m['Destination']}")
