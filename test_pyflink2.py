import sys
sys.path.insert(0, '/opt/flink/opt/python')
sys.path.insert(0, '/opt/flink/opt/python/pyflink.zip')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

try:
    from pyflink.datastream.connectors.jdbc import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions
    print("JDBC OK")
except Exception as e:
    print("JDBC ERROR:", e)

try:
    from pyflink.datastream import StreamExecutionEnvironment
    env = StreamExecutionEnvironment.get_execution_environment()
    print("Env OK, parallelism:", env.get_parallelism())
except Exception as e:
    print("Env ERROR:", e)

print("All imports OK")
