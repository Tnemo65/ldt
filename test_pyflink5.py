import sys
sys.path.insert(0, '/opt/flink/pyflink_extracted')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

import pyflink
from pyflink.datastream import StreamExecutionEnvironment
print("PyFlink OK")
