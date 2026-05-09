import sys
sys.path.insert(0, '/opt/flink/pyflink_extracted')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

from pyflink.datastream import StreamExecutionEnvironment
print("DataStream OK")

from pyflink.datastream.connectors.jdbc import JdbcSink
print("JDBC OK")

from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
print("Kafka OK")

env = StreamExecutionEnvironment.get_execution_environment()
print("Env OK, parallelism:", env.get_parallelism())
