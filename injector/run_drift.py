import subprocess, sys, os, time

script = """
import subprocess, sys

result = subprocess.run(
    ['python3', '/src/drift_injector.py', '--scenario', 'ALL', '--rate', '500'],
    cwd='/src',
    env={'PYTHONPATH': '/usr/local/lib/python3.10/site-packages', 'KAFKA_BOOTSTRAP_SERVERS': 'kafka:9092'},
    capture_output=True, text=True
)
print('STDOUT:', result.stdout[:2000])
print('STDERR:', result.stderr[:1000])
print('RC:', result.returncode)
"""

os.system('docker run --rm --network cadqstream-net -v "c:/proj/ldt:/src:ro" -v "c:/proj/ldt/deployment/kafka:/kafka:ro" python:3.10-slim bash -c "pip install -q kafka-python && python3 -c \\"" + script.replace('"', '\\"') + "\\" " 2>&1')
