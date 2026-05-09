import sys
sys.path.insert(0, "/opt/flink/e2e")
sys.path.insert(0, "/opt/flink/e2e/src")
import features.vectorizer
print("OK")
import inspect
print(inspect.getmembers(features.vectorizer))
