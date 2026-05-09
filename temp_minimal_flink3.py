import os, sys, time
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.common.typeinfo import Types

print("Creating StreamExecutionEnvironment...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)
print(f"Parallelism: {env.get_parallelism()}")

# Simple from_collection
print("Building pipeline...")
ds = env.from_collection([1, 2, 3, 4, 5])
ds.map(lambda x: x * 2).print()

print("Calling env.execute()...")
sys.stdout.flush()
try:
    result = env.execute("Minimal Test")
    print(f"Job finished! Result: {result}")
except Exception as e:
    print(f"Job FAILED: {e}")
    import traceback
    traceback.print_exc()
