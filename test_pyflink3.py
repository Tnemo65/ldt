import sys
sys.path.insert(0, '/opt/flink/opt/python')
sys.path.insert(0, '/opt/flink/opt/python/pyflink.zip')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

print("Python:", sys.version)

try:
    from pyflink.datastream.connectors.jdbc import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions
    print("JDBC: OK")
except Exception as e:
    print("JDBC: ERROR -", e)

try:
    from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
    print("Kafka: OK")
except Exception as e:
    print("Kafka: ERROR -", e)

try:
    from pyflink.common.serialization import SimpleStringSchema
    print("SimpleStringSchema: OK")
except Exception as e:
    print("SimpleStringSchema: ERROR -", e)

try:
    from pyflink.datastream import StreamExecutionEnvironment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    print("StreamExecutionEnvironment: OK, parallelism:", env.get_parallelism())
except Exception as e:
    import traceback
    traceback.print_exc()
    print("StreamExecutionEnvironment: ERROR -", e)
