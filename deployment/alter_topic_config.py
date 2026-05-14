"""
Kafka topic config: alter dq-stream-anomalies cleanup.policy from compact to delete.
Uses kafka-python.
"""
from kafka.admin import KafkaAdminClient, ConfigResource
import time

admin = KafkaAdminClient(
    bootstrap_servers='ldt-kafka:9092',
    client_id='topic-config-alterer'
)

print("=== BEFORE ===")
try:
    cr = ConfigResource('TOPIC', 'dq-stream-anomalies')
    resp = admin.describe_configs([cr])
    print(f"Response type: {type(resp)}")
    print(f"Response content: {resp}")
    for resource, cfg in resp.items():
        print(f"  Resource: {resource}")
        print(f"  Config response type: {type(cfg)}")
        for entry in resource.entries:
            print(f"    {entry.name} = {entry.value}")
except Exception as e:
    print(f"describe_configs error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== ALTERING dq-stream-anomalies ===")
try:
    alter_cr = ConfigResource('TOPIC', 'dq-stream-anomalies',
                              configs={'cleanup.policy': 'delete'})
    result = admin.alter_configs([alter_cr])
    print(f"alter_configs result: {result}")
    time.sleep(3)
except Exception as e:
    print(f"alter_configs error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== AFTER ===")
try:
    cr2 = ConfigResource('TOPIC', 'dq-stream-anomalies')
    resp2 = admin.describe_configs([cr2])
    for resource, cfg in resp2.items():
        for entry in resource.entries:
            if 'cleanup' in entry.name.lower() or 'retention' in entry.name.lower():
                print(f"  {entry.name} = {entry.value}")
except Exception as e:
    print(f"verify error: {e}")
