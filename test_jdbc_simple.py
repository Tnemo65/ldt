import sys
sys.path.insert(0, '/opt/flink/pyflink_extracted')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.datastream.connectors.jdbc import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions
from pyflink.common.typeinfo import Types

print("Testing JDBC connection...")

conn_opts = (
    JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
    .with_url("jdbc:postgresql://postgres:5432/dq_pipeline")
    .with_driver_name("org.postgresql.Driver")
    .with_user_name("cadqstream")
    .with_password("cadqstream123")
    .build()
)

test_type_info = Types.ROW([Types.STRING(), Types.DOUBLE()])
test_sql = "INSERT INTO anomaly_scores (trip_id, anomaly_score) VALUES (?, ?)"
test_sink = JdbcSink.sink(test_sql, test_type_info, conn_opts, JdbcExecutionOptions.builder().with_batch_size(1).with_batch_interval_ms(3000).build())
print("Test sink created: OK")

class TestRow(MapFunction):
    def map(self, value):
        return (str(value), 0.5)

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)

# Create a simple source
from pyflink.datastream import DataStream
ds = env.from_elements(1, 2, 3, 4, 5)
ds.map(TestRow(), output_type=test_type_info).add_sink(test_sink)
print("Sink added to plan, executing...")
env.execute("JDBC-Test")
print("Done!")
