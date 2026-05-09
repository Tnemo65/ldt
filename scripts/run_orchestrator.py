#!/usr/bin/env python3
"""
CA-DQStream E2E Throughput Test Orchestrator (runs inside container)
Usage: python3 /opt/flink/e2e/run_orchestrator.py <rate> <total>
Example: python3 /opt/flink/e2e/run_orchestrator.py 1000 60000
"""
import os, sys, time, json, subprocess

WORK = "/opt/flink/e2e"
LOG = WORK + "/orchestrator.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def exec_cmd(cmd, timeout=30):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def docker_exec(container, cmd, timeout=30):
    return exec_cmd(["docker", "exec", container, "bash", "-c", cmd], timeout=timeout)

# ---- Flink Job Script ----
FLINK_SCRIPT = r"""#!/usr/bin/env python3
import os, sys, time
WORK = "/opt/flink/e2e"
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + "/src")
from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
print("[" + time.strftime("%H:%M:%S") + "] Flink job starting")
sys.stdout.flush()
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
kp = {"bootstrap.servers": "ldt-kafka-1:9092", "group.id": "cadqstream-e2e", "auto.offset.reset": "earliest"}
src = FlinkKafkaConsumer(topics="taxi-nyc-raw", deserialization_schema=SimpleStringSchema(), properties=kp)
stream = env.add_source(src)
class PJ(MapFunction):
    _cnt = 0
    def map(self, v):
        PJ._cnt += 1
        if PJ._cnt % 10000 == 0:
            print("[" + time.strftime("%H:%M:%S") + "] Kafka recv: " + str(PJ._cnt))
            sys.stdout.flush()
        try:
            import json as j
            return j.loads(v)
        except:
            return None
stream = stream.map(PJ(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)
import importlib
kg = importlib.import_module("operators.key_generator")
gen_id = getattr(kg, "generate_trip_id")
dm = importlib.import_module("operators.deduplicator")
DedupCls = getattr(dm, "DeduplicatorFunction")
sm = importlib.import_module("operators.schema_validator")
SchemaCls = getattr(sm, "SchemaValidator")
cm = importlib.import_module("operators.canary_rules")
CanaryCls = getattr(cm, "CanaryRulesValidator")
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
    vm = importlib.import_module("features.vectorizer")
    VecCls = getattr(vm, "FeatureVectorizer")
    im = importlib.import_module("operators.if_scoring_operator")
    gck = getattr(im, "get_context_key")
    import pickle, json
    _m = pickle.load(open(WORK + "/models/iforest_model.pkl", "rb"))
    _s = pickle.load(open(WORK + "/models/scaler.pkl", "rb"))
    _t = json.load(open(WORK + "/models/context_thresholds.json"))
    _v = VecCls()
    print("[" + time.strftime("%H:%M:%S") + "] [ML] Loaded: " + str(_m.n_estimators) + " trees")
    sys.stdout.flush()
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
    print("[" + time.strftime("%H:%M:%S") + "] [ML] Skipped: " + str(e))
    sys.stdout.flush()
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
print("[" + time.strftime("%H:%M:%S") + "] Submitting job (parallelism=" + str(env.get_parallelism()) + ")")
sys.stdout.flush()
result = env.execute("CA-DQStream E2E Test")
print("[" + time.strftime("%H:%M:%S") + "] Job finished: " + str(result))
"""

# ---- Producer Script ----
PRODUCER_SCRIPT = r"""#!/usr/bin/env python3
import json, time, sys
from kafka import KafkaProducer
import pandas as pd
WORK = "/opt/flink/e2e"
OUT = WORK + "/producer.log"
def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")
try:
    producer = KafkaProducer(bootstrap_servers=["ldt-kafka-1:9092"], value_serializer=lambda v: json.dumps(v).encode("utf-8"), retries=5)
except:
    producer = KafkaProducer(bootstrap_servers=["localhost:9092"], value_serializer=lambda v: json.dumps(v).encode("utf-8"), retries=5)
log("[" + time.strftime("%H:%M:%S") + "] Loading parquet data...")
df = pd.read_parquet(WORK + "/data/tripdata.parquet")
log("[" + time.strftime("%H:%M:%S") + "] Loaded " + str(len(df)) + " records")
total = int(sys.argv[1])
rate = int(sys.argv[2])
sleep_t = 1.0 / rate
log("[" + time.strftime("%H:%M:%S") + "] Sending " + str(total) + " at " + str(rate) + "/sec")
sent = 0
start = time.time()
for i in range(total):
    row = df.iloc[i % len(df)]
    record = {}
    for col in df.columns:
        v = row[col]
        if pd.isna(v):
            record[col] = None
        elif hasattr(v, 'item'):
            record[col] = v.item()
        else:
            record[col] = v
        if col in ("tpep_pickup_datetime", "tpep_dropoff_datetime") and record[col]:
            try:
                record[col] = str(pd.Timestamp(record[col]).to_pydatetime())[:19]
            except:
                pass
    try:
        producer.send("taxi-nyc-raw", record)
        sent += 1
        if sent % rate == 0:
            elapsed = time.time() - start
            log("[" + time.strftime("%H:%M:%S") + "] Sent " + str(sent) + "/" + str(total) + " (" + str(round(sent/elapsed, 1)) + "/sec actual)")
        time.sleep(sleep_t)
    except Exception as e:
        log("Error " + str(i) + ": " + str(e))
producer.flush()
producer.close()
elapsed = time.time() - start
log("[" + time.strftime("%H:%M:%S") + "] DONE: " + str(sent) + " sent, " + str(round(elapsed, 1)) + "s total, " + str(round(sent/elapsed, 1)) + "/sec avg")
"""

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    log(f"Wrote {path} ({len(content)} bytes)")

def main():
    rate = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 60000

    log("=" * 60)
    log(f"CA-DQStream E2E Throughput Test (rate={rate}, total={total})")
    log("=" * 60)

    # Write scripts
    log("Writing scripts...")
    write_file(WORK + "/flink_job.py", FLINK_SCRIPT)
    write_file(WORK + "/producer.py", PRODUCER_SCRIPT)
    exec_cmd(["docker", "cp", WORK + "/flink_job.py", f"ldt-flink-jobmanager:{WORK}/flink_job.py"])
    exec_cmd(["docker", "cp", WORK + "/producer.py", f"ldt-flink-jobmanager:{WORK}/producer.py"])
    log("Scripts copied to container")

    # Check Flink
    log("Checking Flink cluster...")
    out, _, rc = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/overview", timeout=15)
    log(f"  Overview: {out}")

    # Clean logs
    exec_cmd(["docker", "exec", "ldt-flink-jobmanager", "bash", "-c", f"rm -f {WORK}/job_out.txt {WORK}/producer.log {WORK}/producer_out.txt"])

    # Start Flink job
    log("Starting Flink job...")
    docker_exec("ldt-flink-jobmanager",
        f"cd {WORK} && rm -f {WORK}/job_out.txt && nohup python3 {WORK}/flink_job.py > {WORK}/job_out.txt 2>&1 &")
    log("  Job submitted, waiting for registration...")

    # Wait for job to register
    job_id = None
    for i in range(15):
        time.sleep(5)
        out, _, _ = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs", timeout=10)
        try:
            jobs = json.loads(out)
            if jobs.get("jobs"):
                job_id = jobs["jobs"][0]["id"]
                state = jobs["jobs"][0]["status"]
                log(f"  Attempt {i+1}: Job {job_id[:8]}... = {state}")
                if state == "RUNNING":
                    break
        except:
            log(f"  Attempt {i+1}: No jobs yet...")
    else:
        log("  WARNING: Job did not reach RUNNING state")

    if job_id:
        # Get job details
        out, _, _ = docker_exec("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{job_id}", timeout=10)
        try:
            jd = json.loads(out)
            log(f"  Job: {jd.get('name')} duration={jd.get('duration', 0)}ms")
        except:
            pass

    # Start producer after short delay
    log(f"Starting producer ({total} events at {rate}/sec)...")
    docker_exec("ldt-flink-jobmanager",
        f"cd {WORK} && rm -f {WORK}/producer_out.txt && nohup python3 {WORK}/producer.py {total} {rate} > {WORK}/producer_out.txt 2>&1 &")
    log("  Producer started")

    # Monitor for duration
    test_duration = total // rate + 120
    log(f"Monitoring for {test_duration}s...")
    check_interval = 20
    for step in range(test_duration // check_interval):
        time.sleep(check_interval)
        elapsed = (step + 1) * check_interval

        # Check job status
        out, _, rc = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs", timeout=10)
        try:
            jobs = json.loads(out)
            for j in jobs.get("jobs", []):
                jid = j["id"]
                jd_out, _, _ = docker_exec("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{jid}", timeout=10)
                jd = json.loads(jd_out)
                log(f"[{elapsed}s] Job {jid[:8]}...: {jd['state']}, {jd['duration']//1000}s")
                for v in jd.get("vertices", []):
                    rm = v["metrics"]
                    log(f"  {v['name'][:50]}: read={rm.get('read-records',0)}, write={rm.get('write-records',0)}")
        except Exception as e:
            log(f"[{elapsed}s] Monitor error: {e}")

        # Check Kafka lag
        out, _, rc = docker_exec("ldt-kafka-1",
            "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1 | head -20",
            timeout=10)
        for line in out.split("\n"):
            if "taxi-nyc-raw" in line or "cadqstream" in line:
                log(f"  Kafka: {line.strip()}")

    # Collect final results
    log("=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)

    log("--- Job Output (last 30) ---")
    out, _, _ = docker_exec("ldt-flink-jobmanager", f"tail -30 {WORK}/job_out.txt")
    log(out)

    log("--- Producer Output ---")
    out, _, _ = docker_exec("ldt-flink-jobmanager", f"cat {WORK}/producer_out.txt")
    log(out)

    log("--- Final Job Metrics ---")
    out, _, _ = docker_exec("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs", timeout=10)
    try:
        jobs = json.loads(out)
        for j in jobs.get("jobs", []):
            jid = j["id"]
            jd_out, _, _ = docker_exec("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{jid}", timeout=10)
            jd = json.loads(jd_out)
            log(f"Job {jid}: {jd['state']}, duration={jd['duration']}ms")
            for v in jd.get("vertices", []):
                rm = v["metrics"]
                log(f"  {v['name'][:60]}: read_records={rm.get('read-records',0)}, write_records={rm.get('write-records',0)}")
    except Exception as e:
        log(f"  Error: {e}")

    log("--- Kafka Lag ---")
    out, _, _ = docker_exec("ldt-kafka-1",
        "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1",
        timeout=10)
    log(out)

    log("=" * 60)
    log("E2E Test Complete!")
    log(f"Full log: {LOG}")
    log("=" * 60)

if __name__ == "__main__":
    main()
