import sys
sys.path.insert(0, '/opt/flink/opt/python')
sys.path.insert(0, '/opt/flink/opt/python/pyflink.zip')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

try:
    from pyflink.datastream import StreamExecutionEnvironment
    print("PyFlink DataStream OK")
except Exception as e:
    print("DataStream ERROR:", e)

try:
    from pyflink.common.serialization import SimpleStringSchema
    print("SimpleStringSchema OK")
except Exception as e:
    print("SimpleStringSchema ERROR:", e)

try:
    from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
    print("Kafka connector OK")
except Exception as e:
    print("Kafka connector ERROR:", e)
