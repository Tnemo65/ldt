import sys
sys.path.insert(0, "/opt/flink/e2e")
sys.path.insert(0, "/opt/flink/e2e/src")
import operators.if_scoring_operator
for name in dir(operators.if_scoring_operator):
    if not name.startswith('_'):
        print(name)
