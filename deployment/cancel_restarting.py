import subprocess

# Cancel via Flink REST API
result = subprocess.run(
    ['docker', 'exec', 'ldt-flink-jobmanager', 'curl', '-s', '-X', 'PATCH',
     'http://localhost:8081/jobs/46038e303f9b77d7c815889023ffde5b/cancel'],
    capture_output=True, text=True
)
print('Cancel:', result.stdout.strip())
