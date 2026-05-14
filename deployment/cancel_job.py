import subprocess

result = subprocess.run(
    ['docker', 'exec', 'ldt-flink-jobmanager', 'curl', '-s', '-X', 'PATCH',
     'http://localhost:8081/jobs/46038e303f9b77d7c815889023ffde5b', '-H', 'Content-Type: application/json',
     '-d', '{"parallelism": 1}'],
    capture_output=True, text=True
)
print('Scale result:', result.stdout.strip())

# Also try cancel
result2 = subprocess.run(
    ['docker', 'exec', 'ldt-flink-jobmanager', 'curl', '-s', '-X', 'PATCH',
     'http://localhost:8081/jobs/46038e303f9b77d7c815889023ffde5b', '-H', 'Content-Type: application/json',
     '-d', '{"type":"CANCEL"}'],
    capture_output=True, text=True
)
print('Cancel result:', result2.stdout.strip())
