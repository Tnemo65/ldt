import os, sys, time
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')
os.environ['PYTHONPATH'] = '/tmp:/tmp/src'

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.common.typeinfo import Types

print("Testing PyFlink environment...")
print("Connecting to Flink cluster...")

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)
print("Environment created")
print(f"Parallelism: {env.get_parallelism()}")

# Simple source
class NumberSource(MapFunction):
    _cnt = 0
    def map(self, value):
        NumberSource._cnt += 1
        if NumberSource._cnt % 1000 == 0:
            print(f"Processed {NumberSource._cnt}")
        return str(NumberSource._cnt)

# Try from_collection
try:
    print("Trying from_collection...")
    ds = env.from_collection([1, 2, 3, 4, 5])
    ds.map(lambda x: x * 2).print()
    print("execute() starting...")
    env.execute("Test Job")
    print("Job finished!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
