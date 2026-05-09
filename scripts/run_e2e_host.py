#!/usr/bin/env python3
"""
CA-DQStream E2E Throughput Test - Host-side orchestrator
Run from host: py scripts/run_e2e_host.py
"""
import subprocess, time, json, sys, os

LOG_FILE = "e2e_host.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def dexp(container, cmd, timeout=30):
    """docker exec"""
    r = subprocess.run(["docker", "exec", container, "bash", "-c", cmd],
                      capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def main():
    log("=" * 60)
    log("CA-DQStream E2E Throughput Test (Host Orchestrator)")
    log("=" * 60)

    WORK = "/opt/flink/e2e"
    rate = 1000
    total = 60000

    # Step 1: Clean up
    log("Step 1: Clean up old processes...")
    dexp("ldt-flink-jobmanager", "pkill -9 -f flink_job.py 2>/dev/null; pkill -9 -f producer.py 2>/dev/null; echo cleaned")

    # Step 2: Cancel any existing Flink jobs
    log("Step 2: Cancel existing Flink jobs...")
    out, _, _ = dexp("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs")
    try:
        for j in json.loads(out).get("jobs", []):
            jid = j["id"]
            log(f"  Cancelling {jid[:8]}...")
            dexp("ldt-flink-jobmanager", f"flink cancel {jid} -m ldt-flink-jobmanager:8081", timeout=15)
    except:
        pass
    time.sleep(5)

    # Step 3: Reset Kafka offsets
    log("Step 3: Reset Kafka offsets...")
    dexp("ldt-kafka-1",
        "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --topic taxi-nyc-raw --reset-offsets --to-earliest --execute",
        timeout=10)

    # Step 4: Clean logs
    log("Step 4: Clean logs...")
    dexp("ldt-flink-jobmanager", f"rm -f {WORK}/job_out.txt {WORK}/producer_out.txt")

    # Step 5: Verify Flink cluster
    log("Step 5: Verify Flink cluster...")
    out, _, _ = dexp("ldt-flink-jobmanager", "curl -s http://localhost:8081/overview")
    try:
        ov = json.loads(out)
        log(f"  TMs={ov['taskmanagers']}, slots={ov['slots-total']}/{ov['slots-available']}")
    except:
        log(f"  Flink check failed: {out}")
        return

    # Step 6: Start Flink job
    log("Step 6: Start Flink job...")
    dexp("ldt-flink-jobmanager",
        f"cd {WORK} && nohup python3 {WORK}/flink_job.py > {WORK}/job_out.txt 2>&1 &")
    log("  Job submitted, waiting...")

    # Wait for job to register
    job_id = None
    for i in range(12):
        time.sleep(5)
        out, _, _ = dexp("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs", timeout=10)
        try:
            jobs = json.loads(out)
            if jobs.get("jobs"):
                job_id = jobs["jobs"][0]["id"]
                state = jobs["jobs"][0]["status"]
                log(f"  Check {i+1}: {job_id[:8]}... = {state}")
                if state == "RUNNING":
                    break
        except:
            log(f"  Check {i+1}: no jobs / parse error")
    else:
        log("  WARNING: Job did not reach RUNNING state")

    # Step 7: Start producer
    log(f"Step 7: Start producer ({total} events at {rate}/sec)...")
    dexp("ldt-flink-jobmanager",
        f"cd {WORK} && nohup python3 {WORK}/producer.py {total} {rate} > {WORK}/producer_out.txt 2>&1 &")

    # Step 8: Monitor
    test_duration = total // rate + 120
    log(f"Step 8: Monitoring for {test_duration}s...")
    for step in range(test_duration // 20):
        time.sleep(20)
        elapsed = (step + 1) * 20

        # Check job
        out, _, _ = dexp("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs", timeout=10)
        try:
            jobs = json.loads(out)
            for j in jobs.get("jobs", []):
                jid = j["id"]
                jd_out, _, _ = dexp("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{jid}", timeout=10)
                jd = json.loads(jd_out)
                log(f"[{elapsed}s] Job {jid[:8]}...: {jd['state']}, {jd['duration']//1000}s")
                for v in jd.get("vertices", []):
                    rm = v["metrics"]
                    log(f"  {v['name'][:50]}: read={rm.get('read-records',0)}, write={rm.get('write-records',0)}")
        except Exception as e:
            log(f"[{elapsed}s] Monitor error: {e}")

        # Check Kafka lag
        out, _, _ = dexp("ldt-kafka-1",
            "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1 | head -10",
            timeout=10)
        for line in out.split("\n"):
            if "taxi-nyc-raw" in line:
                log(f"  Kafka: {line.strip()}")

    # Step 9: Collect results
    log("=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)

    log("--- Job Output (last 30 lines) ---")
    out, _, _ = dexp("ldt-flink-jobmanager", f"tail -30 {WORK}/job_out.txt")
    log(out)

    log("--- Producer Output ---")
    out, _, _ = dexp("ldt-flink-jobmanager", f"cat {WORK}/producer_out.txt")
    log(out)

    log("--- Final Job Metrics ---")
    out, _, _ = dexp("ldt-flink-jobmanager", "curl -s http://localhost:8081/jobs", timeout=10)
    try:
        for j in json.loads(out).get("jobs", []):
            jid = j["id"]
            jd_out, _, _ = dexp("ldt-flink-jobmanager", f"curl -s http://localhost:8081/jobs/{jid}", timeout=10)
            jd = json.loads(jd_out)
            log(f"Job {jid}: {jd['state']}, duration={jd['duration']}ms")
            for v in jd.get("vertices", []):
                rm = v["metrics"]
                log(f"  {v['name'][:60]}: read={rm.get('read-records',0)}, write={rm.get('write-records',0)}")
    except Exception as e:
        log(f"Error: {e}")

    log("--- Kafka Lag ---")
    out, _, _ = dexp("ldt-kafka-1",
        "kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream-e2e --describe 2>&1",
        timeout=10)
    log(out)

    log("=" * 60)
    log("E2E Test Complete!")
    log("=" * 60)

if __name__ == "__main__":
    main()
