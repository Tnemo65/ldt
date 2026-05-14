import subprocess, sys, os

script = 'import sys; print(sys.path); from kafka import KafkaProducer; p = KafkaProducer(bootstrap_servers=["kafka:9092"]); print("OK"); p.close()'

result = subprocess.run(
    ['docker', 'run', '--rm', '--network', 'cadqstream-net', '-v', 'c:/proj/ldt:/src:ro', 'python:3.10-slim',
     'bash', '-c', f'pip install -q kafka-python && python3 -c "{script}"'],
    capture_output=True, text=True, timeout=30
)
print('STDOUT:', result.stdout[:1000])
print('STDERR:', result.stderr[:1000])
print('RC:', result.returncode)
