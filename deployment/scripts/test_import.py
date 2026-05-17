import sys
sys.path.insert(0, '/opt/flink/e2e')
import src.flink_job_complete as m
print('flink_job_complete imported OK')
te = hasattr(m, 'TumblingEventTimeWindows')
print('TumblingEventTimeWindows:', te)
