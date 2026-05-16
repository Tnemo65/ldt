import json, subprocess

def cmd(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout

JID = "b4424487156305660b260e8d66c48a38"
d = json.loads(cmd(["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
    f"http://localhost:8081/jobs/{JID}"]))

print(f"Job State: {d['state']} Duration: {d['duration']/1000:.1f}s")
for i, v in enumerate(d['vertices']):
    vid = v['id'][:8]
    wr = v['metrics']['write-records']
    rr = v['metrics']['read-records']
    state = v['status']
    print(f"  V{i+1}({vid}): read={rr:6d} write={wr:6d} status={state}")

cp_data = cmd(["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
    f"http://localhost:8081/jobs/{JID}/checkpoints"])
cp = json.loads(cp_data)
h = cp.get('history', [])
print(f"\nCheckpoints: total={cp['counts']['total']} completed={cp['counts']['completed']}")
if h:
    last = h[-1]
    print(f"  Latest: id={last['id']} status={last['status']} acks={last['num_acknowledged_subtasks']}/{last['num_subtasks']} type={last.get('checkpoint_type','N/A')}")
