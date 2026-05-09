import os, sys, time, threading
sys.path.insert(0, '/opt/flink/e2e')
sys.path.insert(0, '/opt/flink/e2e/src')
os.environ['PYTHONPATH'] = '/opt/flink/e2e:/opt/flink/e2e/src'

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.common.typeinfo import Types

print("Creating environment...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
print(f"Parallelism: {env.get_parallelism()}")

# Simple pipeline
class Source(MapFunction):
    _cnt = 0
    def map(self, value):
        Source._cnt += 1
        if Source._cnt % 100 == 0:
            print(f"Processed {Source._cnt}")
        return str(Source._cnt)

ds = env.from_collection([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ds.map(Source()).print()

print("Starting job in background thread...")
sys.stdout.flush()

def run_job():
    try:
        result = env.execute("Threaded Test")
        print(f"Job finished: {result}")
    except Exception as e:
        print(f"Job error: {e}")
        import traceback
        traceback.print_exc()

t = threading.Thread(target=run_job, daemon=False)
t.start()

# Wait a bit then exit
print("Main thread exiting in 5 seconds...")
sys.stdout.flush()
time.sleep(5)
print("Main thread done!")
