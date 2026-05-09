import subprocess
r = subprocess.run(["docker", "exec", "ldt-kafka", "env"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "KAFKA" in line:
        print(line)
