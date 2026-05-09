#!/usr/bin/env python3
"""
CA-DQStream E2E Throughput Test Orchestrator
Run from host: py scripts/run_e2e_throughput.py
"""
import subprocess, time, sys, os, json

def docker_exec(container, cmd, timeout=30):
    r = subprocess.run(["docker", "exec", container, "bash", "-c", cmd],
                      capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def docker_cp(src, dst):
    r = subprocess.run(["docker", "cp", src, dst], capture_output=True, text=True, timeout=30)
    return r.returncode

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

WORK_DIR = "/opt/flink/e2e"
LOG_FILE = "e2e_throughput.log"

FLINK_JOB = r"""#!/usr/bin/env python3
import os, sys, time
WORK = "/opt/flink/e2e"
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + "/src")
from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
print(f"[{time.strftime('%H:%M:%S')}] Flink job starting")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
kafka_props = {"bootstrap.servers": "ldt-kafka-1:9092", "group.id": "cadqstream-e2e", "auto.offset.reset": "earliest"}
kafka_source = FlinkKafkaConsumer(topics="taxi-nyc-raw", deserialization_schema=SimpleStringSchema(), properties=kafka_props)
stream = env.add_source(kafka_source)
class PJ(MapFunction):
    _cnt = 0
    def map(self, v):
        PJ._cnt += 1
        if PJ._cnt % 10000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Kafka recv: {PJ._cnt}")
        try:
            import json as j
            return j.loads(v)
        except:
            return None
stream = stream.map(PJ(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)
import importlib
kg = importlib.import_module("operators.key_generator")
gen_id = getattr(kg, "generate_trip_id")
dedup_mod = importlib.import_module("operators.deduplicator")
DedupCls = getattr(dedup_mod, "DeduplicatorFunction")
schema_mod = importlib.import_module("operators.schema_validator")
SchemaCls = getattr(schema_mod, "SchemaValidator")
canary_mod = importlib.import_module("operators.canary_rules")
CanaryCls = getattr(canary_mod, "CanaryRulesValidator")
class AT(MapFunction):
    def map(self, r):
        if r:
            r["trip_id"] = gen_id(r)
        return r
stream = stream.map(AT(), output_type=Types.PICKLED_BYTE_ARRAY())
deduped = stream.key_by(lambda x: x.get("trip_id", ""), key_type=Types.STRING()).map(DedupCls(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)
validator = SchemaCls()
valid = deduped.filter(validator)
canary = valid.map(CanaryCls(), output_type=Types.PICKLED_BYTE_ARRAY())
try:
    vec_mod = importlib.import_module("features.vectorizer")
    VecCls = getattr(vec_mod, "FeatureVectorizer")
    ifs_mod = importlib.import_module("operators.if_scoring_operator")
    gck = getattr(ifs_mod, "get_context_key")
    import pickle, json
    _m = pickle.load(open(WORK + "/models/iforest_model.pkl", "rb"))
    _s = pickle.load(open(WORK + "/models/scaler.pkl", "rb"))
    _t = json.load(open(WORK + "/models/context_thresholds.json"))
    _v = VecCls()
    print(f"[{time.strftime('%H:%M:%S')}] [ML] Loaded: {_m.n_estimators} trees")
    class ML(MapFunction):
        _cnt = 0
        def map(self, r):
            if not r:
                return None
            ML._cnt += 1
            try:
                f = _v.transform(r)
                fs = _s.transform([f])[0]
                raw = _m.score_samples(fs.reshape(1, -1))[0]
                score = -raw
                ctx = gck(r)
                thr = _t.get("thresholds", {}).get(ctx, _t.get("global_threshold", 0.5))
                return dict(**r, anomaly_score=float(score), threshold=float(thr), is_anomaly=bool(score > thr), context_key=ctx)
            except:
                return dict(**r, anomaly_score=0.5, is_anomaly=False, context_key="err")
    ml = valid.map(ML(), output_type=Types.PICKLED_BYTE_ARRAY())
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ML] Skipped: {e}")
    import random
    class Mock(MapFunction):
        def map(self, r):
            if not r:
                return None
            return dict(**r, anomaly_score=random.uniform(0.2, 0.8), is_anomaly=False, context_key="mock")
    ml = valid.map(Mock(), output_type=Types.PICKLED_BYTE_ARRAY())
valid.filter(lambda x: x).print()
canary.filter(lambda x: x).print()
ml.filter(lambda x: x and x.get("is_anomaly")).print()
print(f"[{time.strftime('%H:%M:%S')}] Submitting job (parallelism={env.get_parallelism()})")
sys.stdout.flush()
result = env.execute("CA-DQStream E2E Test")
print(f"[{time.strftime('%H:%M:%S')}] Job finished: {result}")
"""

PRODUCER = r"""#!/usr/bin/env python3
import json, time, sys
from kafka import KafkaProducer
WORK = "/opt/flink/e2e"
OUT = WORK + "/producer.log"
def log(msg):
    with open(OUT, "a") as f:
        f.write(str(msg) + "\n")
try:
    producer = KafkaProducer(bootstrap_servers=["ldt-kafka-1:9092"], value_serializer=lambda v: json.dumps(v).encode("utf-8"), retries=5)
except:
    producer = KafkaProducer(bootstrap_servers=["localhost:9092"], value_serializer=lambda v: json.dumps(v).encode("utf-8"), retries=5)
log(f"[{time.strftime('%H:%M:%S')}] Producer starting, loading data...")
data = []
for path in [WORK + "/src/../data/nyc_taxi_sample.json", "/data/nyc_taxi_sample.json", WORK + "/nyc_taxi_sample.json"]:
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        if data:
            break
    except:
        pass
if not data:
    log("ERROR: No data loaded")
    sys.exit(1)
log(f"[{time.strftime('%H:%M:%S')}] Loaded {len(data)} records")
total = int(sys.argv[1])
rate = int(sys.argv[2])
sleep_t = 1.0 / rate
log(f"[{time.strftime('%H:%M:%S')}] Sending {total} at {rate}/sec")
sent = 0
start = time.time()
for i in range(total):
    try:
        producer.send("taxi-nyc-raw", data[i % len(data)])
        sent += 1
        if i > 0 and i % rate == 0:
            elapsed = time.time() - start
            log(f"[{time.strftime('%H:%M:%S')}] Sent {sent}/{total} ({sent/elapsed:.1f}/sec)")
        time.sleep(sleep_t)
    except Exception as e:
        log(f"Error {i}: {e}")
producer.flush()
producer.close()
elapsed = time.time() - start
log(f"[{time.strftime('%H:%M:%S')}] Done: {sent} sent, {elapsed:.1f}s total, {sent/elapsed:.1f}/sec avg")
"""

def write_file_in_container(container, path, content):
    encoded = content.replace("\\", "\\\\").replace("'", "'\"'\"'").replace("\n", "'$'\\n'")
    cmd = f"python3 -c \"open('{path}','w').write('" + encoded + "')\""
    docker_exec(container, cmd, timeout=10)

def main():
    global LOG_FILE
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_throughput.log")
    log("=" * 60)
    log("CA-DQStream E2E Throughput Test")
    log("=" * 60)

    # Step 1: Cancel existing jobs
    log("Step 1: Cancel existing Flink jobs...")
    out, err, rc = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs")
    try:
        jobs = json.loads(out)
        for j in jobs.get("jobs", []):
            jid = j["id"]
            log(f"  Cancelling {jid}...")
            docker_exec("ldt-flink-jobmanager", f"curl -s -X PATCH http://localhost:8081/jobs/{jid}/messages -H 'Content-Type: application/json' -d '{{\"type\":\"CANCEL_JOB\"}}'", timeout=15)
    except:
        log(f"  No jobs or error: {out}")
    time.sleep(5)

    # Step 2: Reset Kafka offsets
    log("Step 2: Reset Kafka consumer group offsets...")
    for group in ["cadqstream-e2e", "cadqstream-flink-consumer"]:
        for topic in ["taxi-nyc-raw"]:
            out, err, rc = docker_exec("ldt-kafka-1",
                f"kafka-consumer-groups --bootstrap-server localhost:9092 --group {group} --topic {topic} --reset-offsets --to-earliest --execute",
                timeout=15)
            if rc == 0:
                log(f"  Reset {group}/{topic}: {out}")
            else:
                log(f"  Reset {group}/{topic}: {err[:100]}")

    # Step 3: Check infrastructure
    log("Step 3: Check Flink cluster...")
    out, err, rc = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/overview")
    log(f"  {out}")

    # Step 4: Write scripts to container
    log("Step 4: Write scripts to container...")
    write_file_in_container("ldt-flink-jobmanager", WORK_DIR + "/flink_job.py", FLINK_JOB)
    write_file_in_container("ldt-flink-jobmanager", WORK_DIR + "/producer.py", PRODUCER)
    log("  Scripts written")

    # Step 5: Clean up old logs
    docker_exec("ldt-flink-jobmanager", f"rm -f {WORK_DIR}/job.log {WORK_DIR}/job_out.txt {WORK_DIR}/producer.log {WORK_DIR}/producer_out.txt {WORK_DIR}/monitor_out.txt")

    # Step 6: Start Flink job
    log("Step 5: Start Flink job in background...")
    docker_exec("ldt-flink-jobmanager", f"cd {WORK_DIR} && nohup python3 {WORK_DIR}/flink_job.py > {WORK_DIR}/job_out.txt 2>&1 &")
    log("  Job submitted")

    # Wait for job to register
    log("  Waiting for job to register (up to 60s)...")
    for i in range(12):
        time.sleep(5)
        out, err, rc = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs")
        try:
            jobs = json.loads(out)
            if jobs.get("jobs"):
                jid = jobs["jobs"][0]["id"]
                state = jobs["jobs"][0]["status"]
                log(f"  Job {jid[:8]}... is {state}")
                if state == "RUNNING":
                    break
        except:
            pass

    # Step 7: Start producer
    target_rate = 1000
    total_events = 60000
    log(f"Step 6: Start producer ({total_events} events at {target_rate}/sec)...")
    docker_exec("ldt-flink-jobmanager", f"cd {WORK_DIR} && nohup python3 {WORK_DIR}/producer.py {total_events} {target_rate} > {WORK_DIR}/producer_out.txt 2>&1 &")
    log("  Producer started")

    # Step 8: Monitor for duration
    test_duration = total_events // target_rate + 60
    log(f"Step 7: Monitoring for {test_duration}s...")
    time.sleep(30)
    for i in range((test_duration - 30) // 15):
        time.sleep(15)
        out, err, rc = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs")
        try:
            jobs = json.loads(out)
            for j in jobs.get("jobs", []):
                jid = j["id"]
                jd_out, _, _ = docker_exec("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{jid}")
                jd = json.loads(jd_out)
                log(f"  Job {jid[:8]}...: state={jd['state']}, duration={jd['duration']//1000}s")
                for v in jd.get("vertices", []):
                    rm = v["metrics"]
                    log(f"    {v['name'][:40]}: read={rm.get('read-records',0)}, write={rm.get('write-records',0)}")
        except Exception as e:
            log(f"  Monitor error: {e}")

        out, err, rc = docker_exec("ldt-kafka-1", "kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>&1 | head -10")
        log(f"  Kafka: {out}")

    # Step 9: Collect final results
    log("Step 8: Collect results...")
    log("--- Job Output (last 30 lines) ---")
    out, _, _ = docker_exec("ldt-flink-jobmanager", f"tail -30 {WORK_DIR}/job_out.txt")
    log(out)

    log("--- Producer Output ---")
    out, _, _ = docker_exec("ldt-flink-jobmanager", f"cat {WORK_DIR}/producer_out.txt")
    log(out)

    log("--- Final Flink Metrics ---")
    out, _, _ = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs")
    try:
        jobs = json.loads(out)
        for j in jobs.get("jobs", []):
            jid = j["id"]
            jd_out, _, _ = docker_exec("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{jid}")
            jd = json.loads(jd_out)
            log(f"  Job {jid}: state={jd['state']}, duration={jd['duration']}ms")
            for v in jd.get("vertices", []):
                rm = v["metrics"]
                log(f"    {v['name'][:60]}")
                log(f"      read_records={rm.get('read-records', 0)}, write_records={rm.get('write-records', 0)}")
    except:
        pass

    log("--- Kafka Lag ---")
    out, _, _ = docker_exec("ldt-kafka-1", "kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>&1 | head -20")
    log(out)

    log("=" * 60)
    log("E2E Throughput Test Complete!")
    log("=" * 60)
    log(f"Full log: {LOG_FILE}")

if __name__ == "__main__":
    main()
