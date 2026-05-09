import os, sys, time
sys.path.insert(0, '/opt/flink/e2e')
sys.path.insert(0, '/opt/flink/e2e/src')
os.environ['PYTHONPATH'] = '/opt/flink/e2e:/opt/flink/e2e/src'
os.environ['FLINK_REST_ADDR'] = 'http://localhost:8081'

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.common.typeinfo import Types

print("Testing RPC connection...")
import requests
try:
    r = requests.get("http://localhost:8081/overview", timeout=5)
    print(f"  REST works: {r.status_code}")
except Exception as e:
    print(f"  REST failed: {e}")

print("Creating environment...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
print(f"Parallelism: {env.get_parallelism()}")

# Try disabling Python UDF execution mode
from pyflink.table import EnvironmentSettings, BatchTableEnvironment
from pyflink.datastream import StreamExecutionEnvironment

# Check if there's a way to set the REST address
print("Environment config:")
print(f"  REST URL should be: http://flink-jobmanager:8081")

# Simple pipeline
class Source(MapFunction):
    _cnt = 0
    def map(self, value):
        Source._cnt += 1
        if Source._cnt % 100 == 0:
            print(f"Processed {Source._cnt}")
        return str(Source._cnt)

ds = env.from_collection([1, 2, 3, 4, 5])
ds.map(Source()).print()

print("execute() starting...")
sys.stdout.flush()
try:
    result = env.execute("Minimal Test")
    print(f"Job finished: {result}")
except Exception as e:
    print(f"Job FAILED: {e}")
    import traceback
    traceback.print_exc()
