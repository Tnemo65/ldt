import os, sys, time, requests
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

# Check REST connectivity first
print("Testing REST connectivity...")
try:
    r = requests.get("http://localhost:8081/overview", timeout=5)
    print(f"  /overview: {r.status_code} - {r.json()}")
except Exception as e:
    print(f"  /overview FAILED: {e}")

try:
    r = requests.get("http://localhost:8081/jobs", timeout=5)
    print(f"  /jobs: {r.status_code} - {r.json()}")
except Exception as e:
    print(f"  /jobs FAILED: {e}")

# Test PyFlink
from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.common.typeinfo import Types

print("\nCreating StreamExecutionEnvironment...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)
print(f"Parallelism: {env.get_parallelism()}")

# Build a minimal pipeline
class NumSource(MapFunction):
    _cnt = 0
    def map(self, value):
        NumSource._cnt += 1
        if NumSource._cnt % 1000 == 0:
            print(f"  Source: {NumSource._cnt}")
        return str(NumSource._cnt)

# Simpler: just count
ds = env.from_parallel_collection(Types.INT(), parallelism=1)
ds.map(NumSource()).print()

print("\nCalling env.execute()...")
sys.stdout.flush()
try:
    result = env.execute("Minimal Test")
    print(f"\nJob finished! Result: {result}")
except Exception as e:
    print(f"\nJob FAILED: {e}")
    import traceback
    traceback.print_exc()
