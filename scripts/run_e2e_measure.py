#!/usr/bin/env python3
"""
CA-DQStream E2E Throughput Test
Run: py scripts/run_e2e_measure.py

Steps:
1. Reset Kafka offsets
2. Start Flink job
3. Flood Kafka with 100K synthetic trips
4. Measure Flink throughput via Kafka lag depletion
5. Report results
"""
import subprocess, time, json, sys, os

LOG = "e2e_measure.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def dexec(container, cmd, timeout=30):
    r = subprocess.run(["docker", "exec", container, "bash", "-c", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def dexec_bg(container, cmd):
    subprocess.Popen(["docker", "exec", container, "bash", "-c", cmd + " >/dev/null 2>&1"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

WORK = "/opt/flink/e2e"

# ---- Synthetic data generator (no pandas needed) ----
SYNTH_PRODUCER = r"""#!/usr/bin/env python3
import json, time, sys, random
from kafka import KafkaProducer
WORK = "/opt/flink/e2e"
OUT = WORK + "/synth.log"

def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")

VENDOR_IDS = [1, 2]
RATECODE_IDS = [1, 2, 3, 4, 5, 6, 99]
PULOCS = list(range(1, 266))
DOLOCS = list(range(1, 266))
PASSCNTS = [1, 2, 3, 4, 5, 6]

def hour_label(h):
    if 7 <= h < 10: return "morning_rush"
    elif 10 <= h < 16: return "midday"
    elif 16 <= h < 20: return "evening_rush"
    else: return "night"

def zone_name(loc):
    if loc <= 12: return "airport"
    elif loc <= 74: return "manhattan"
    elif loc <= 199: return "brooklyn"
    elif loc <= 229: return "queens"
    elif loc <= 246: return "bronx"
    else: return "staten_island"

def is_weekend(day):
    return day % 7 >= 5

def gen_trip(day, hour):
    dur = random.uniform(0.05, 1.0)
    dist = random.uniform(0.3, 25.0)
    fare = round(random.uniform(3.0, 80.0), 2)
    speed = dist / dur if dur > 0 else 0
    extra = round(random.uniform(0, 3.5), 2)
    tip = round(fare * random.uniform(0, 0.25), 2) if random.random() > 0.3 else 0
    tolls = round(random.uniform(0, 20), 2) if random.random() > 0.9 else 0
    mta = 0.5
    imp = 1.0
    cong = 2.5
    pu_dt = f"2024-01-{day:02d} {hour:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
    dur_s = int(dur * 3600)
    do_m, do_s = (random.randint(0,59), random.randint(0,59))
    do_h = hour + int(dur_s // 3600)
    do_d = day + int(do_h // 24)
    do_dt = f"2024-01-{do_d:02d} {do_h % 24:02d}:{do_m:02d}:{do_s:02d}"
    return {
        "VendorID": random.choice(VENDOR_IDS),
        "tpep_pickup_datetime": pu_dt,
        "tpep_dropoff_datetime": do_dt,
        "passenger_count": float(random.choice(PASSCNTS)),
        "trip_distance": round(dist, 2),
        "RatecodeID": float(random.choice(RATECODE_IDS)),
        "store_and_fwd_flag": random.choice(["N", "Y"]),
        "PULocationID": float(random.choice(PULOCS)),
        "DOLocationID": float(random.choice(DOLOCS)),
        "payment_type": float(random.choice([1, 2, 3, 4, 5, 6])),
        "fare_amount": fare,
        "extra": extra,
        "mta_tax": mta,
        "tip_amount": tip,
        "tolls_amount": tolls,
        "improvement_surcharge": imp,
        "total_amount": round(fare + extra + mta + tip + tolls + imp + cong, 2),
        "congestion_surcharge": cong,
        "Airport_fee": 2.5,
        "trip_duration": round(dur, 6),
        "speed_mph": round(speed, 6),
    }

try:
    producer = KafkaProducer(
        bootstrap_servers=["ldt-kafka-1:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
        batch_size=131072,
        linger_ms=20,
        buffer_memory=268435456,
        compression_type="gzip",
    )
except:
    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
    )

total = int(sys.argv[1])
log("[" + time.strftime("%H:%M:%S") + "] Starting: " + str(total) + " events")
sent = 0
start = time.time()
# Generate realistic hours/days
hours = [i % 24 for i in range(total // 50 + 1)]
days = [1 + (i // 24) for i in range(total // 50 + 1)]
for i in range(total):
    trip = gen_trip(days[i % len(days)], hours[i % len(hours)])
    producer.send("taxi-nyc-raw", trip)
    sent += 1
    if sent % 50000 == 0:
        elapsed = time.time() - start
        log("[" + time.strftime("%H:%M:%S") + "] " + str(sent) + "/" + str(total) + " (" + str(round(sent/elapsed)) + "/sec)")
producer.flush()
producer.close()
elapsed = time.time() - start
log("[" + time.strftime("%H:%M:%S") + "] DONE: " + str(sent) + " in " + str(round(elapsed, 1)) + "s = " + str(round(sent/elapsed)) + "/sec avg")
"""

# ---- Flink Job (with correct get_context_key call) ----
FLINK_JOB = r"""#!/usr/bin/env python3
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
        if PJ._cnt % 20000 == 0:
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
im = importlib.import_module("operators.if_scoring_operator")
gck = getattr(im, "get_context_key")

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
    import pickle, json
    _m = pickle.load(open(WORK + "/models/iforest_model.pkl", "rb"))
    _s = pickle.load(open(WORK + "/models/scaler.pkl", "rb"))
    _t = json.load(open(WORK + "/models/context_thresholds.json"))
    _v = VecCls()
    print("[" + time.strftime("%H:%M:%S") + "] [ML] Loaded: " + str(_m.n_estimators) + " trees, thresholds=" + str(len(_t.get("thresholds", {}))))
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
            except Exception as e:
                return dict(**r, anomaly_score=0.5, is_anomaly=False, context_key="err:" + str(e)[:20])
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
print("[" + time.strftime("%H:%M:%S") + "] Starting execute() (parallelism=" + str(env.get_parallelism()) + ")")
sys.stdout.flush()
result = env.execute("CA-DQStream E2E Test")
print("[" + time.strftime("%H:%M:%S") + "] Job finished: " + str(result))
"""

def write_and_copy(local_path, content, container):
    with open(local_path, "w") as f:
        f.write(content)
    r = subprocess.run(["docker", "cp", local_path, f"{container}:{WORK}/"], capture_output=True, text=True)
    return r.returncode == 0

def main():
    total_events = 100000

    log("=" * 60)
    log(f"CA-DQStream E2E Throughput Test ({total_events} events)")
    log("=" * 60)

    # Step 1: Kill any old jobs and clean
    log("Step 1: Clean up...")
    dexec("ldt-flink-jobmanager", "pkill -9 -f flink_job.py 2>/dev/null; pkill -9 -f producer.py 2>/dev/null; pkill -9 -f synth.py 2>/dev/null; echo clean")

    # Step 2: Reset Kafka offsets
    log("Step 2: Reset Kafka offsets...")
    dexec("ldt-kafka-1",
        "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --topic taxi-nyc-raw --reset-offsets --to-earliest --execute",
        timeout=15)

    # Step 3: Check cluster
    log("Step 3: Check Flink cluster...")
    out, _, rc = dexec("ldt-flink-jobmanager", "curl -s http://localhost:8081/overview", timeout=10)
    try:
        ov = json.loads(out)
        log(f"  TMs={ov['taskmanagers']}, slots={ov['slots-total']}/{ov['slots-available']}")
    except:
        log(f"  Flink not ready: {out}")

    # Step 4: Write scripts
    log("Step 4: Write scripts to container...")
    write_and_copy(WORK + "/synth_producer.py", SYNTH_PRODUCER, "ldt-flink-jobmanager")
    write_and_copy(WORK + "/flink_job2.py", FLINK_JOB, "ldt-flink-jobmanager")
    dexec("ldt-flink-jobmanager", f"rm -f {WORK}/synth.log {WORK}/job_out2.txt {WORK}/job_out.txt")
    log("  Scripts ready")

    # Step 5: Start Flink job
    log("Step 5: Start Flink job...")
    dexec_bg("ldt-flink-jobmanager",
        f"cd {WORK} && nohup python3 {WORK}/flink_job2.py > {WORK}/job_out2.txt 2>&1")

    # Wait for job to start (via job output)
    log("  Waiting for job to start...")
    started = False
    for i in range(20):
        time.sleep(5)
        out, _, _ = dexec("ldt-flink-jobmanager", f"grep -c 'Kafka recv\\|Starting execute' {WORK}/job_out2.txt 2>/dev/null || echo 0", timeout=5)
        try:
            cnt = int(out)
            if cnt > 0:
                started = True
                break
        except:
            pass
        log(f"  Check {i+1}/20: waiting...")

    if started:
        log("  Job is running!")
    else:
        log("  WARNING: Job may not have started properly")
        out, _, _ = dexec("ldt-flink-jobmanager", f"tail -10 {WORK}/job_out2.txt", timeout=5)
        log(f"  Last output: {out}")

    # Step 6: Flood Kafka with data
    log("Step 6: Flood Kafka with synthetic data...")
    dexec("ldt-flink-jobmanager", f"rm -f {WORK}/synth.log")
    dexec_bg("ldt-flink-jobmanager",
        f"cd {WORK} && python3 {WORK}/synth_producer.py {total_events} > /dev/null 2>&1")

    # Step 7: Monitor throughput via Kafka lag
    log("Step 7: Monitor throughput...")
    log("  Waiting for data to flood in...")
    time.sleep(10)  # Let producer flood

    for i in range(20):
        time.sleep(10)
        elapsed = (i + 1) * 10

        # Check producer status
        out, _, _ = dexec("ldt-flink-jobmanager", f"tail -3 {WORK}/synth.log 2>/dev/null || echo 'no log'")
        log(f"[{elapsed}s] Producer: {out}")

        # Check Kafka lag
        out, _, _ = dexec("ldt-kafka-1",
            "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1",
            timeout=10)
        for line in out.split("\n"):
            if "taxi-nyc-raw" in line:
                parts = line.split()
                if len(parts) >= 8:
                    lag = parts[7]
                    log(f"  Kafka lag: {lag}")

        # Check job output count
        out, _, _ = dexec("ldt-flink-jobmanager",
            f"grep -c 'Kafka recv' {WORK}/job_out2.txt 2>/dev/null || echo 0",
            timeout=5)
        log(f"  Job recv count: {out}")

    # Step 8: Final results
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)

    log("--- Job Output (last 20 lines) ---")
    out, _, _ = dexec("ldt-flink-jobmanager", f"tail -20 {WORK}/job_out2.txt")
    log(out)

    log("--- Producer Summary ---")
    out, _, _ = dexec("ldt-flink-jobmanager", f"grep DONE {WORK}/synth.log 2>/dev/null || cat {WORK}/synth.log | tail -3")
    log(out)

    log("--- Kafka Lag ---")
    out, _, _ = dexec("ldt-kafka-1",
        "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1",
        timeout=10)
    log(out)

    log("--- Kafka Log End Offsets ---")
    out, _, _ = dexec("ldt-kafka-1",
        "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1",
        timeout=10)
    for line in out.split("\n"):
        if "taxi-nyc-raw" in line:
            parts = line.split()
            if len(parts) >= 8:
                log(f"  Partition {parts[1]}: offset={parts[2]} leo={parts[4]} lag={parts[7]}")

    log("=" * 60)
    log("Test complete!")
    log("=" * 60)
    log(f"Full log: {LOG}")

if __name__ == "__main__":
    main()
