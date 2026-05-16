import json, subprocess

def cmd(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout

JID = "9d934b955a9cc91b802653daf912fe52"
d = json.loads(cmd(["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
    f"http://localhost:8081/jobs/{JID}"]))

print(f"\nJob: {d['jid']} State: {d['state']} Duration: {d['duration']/1000:.1f}s")
for i, v in enumerate(d['vertices']):
    vid = v['id'][:8]
    wr = v['metrics']['write-records']
    rr = v['metrics']['read-records']
    state = v['status']
    print(f"  V{i+1}({vid}): read={rr:6d} write={wr:6d} status={state}")

cp = json.loads(cmd(["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
    f"http://localhost:8081/jobs/{JID}/checkpoints"]))
h = cp.get('history', [])
if h:
    last = h[-1]
    print(f"\nCheckpoint: id={last['id']} status={last['status']} acks={last['num_acknowledged_subtasks']}/{last['num_subtasks']} type={last.get('checkpoint_type','N/A')}")
