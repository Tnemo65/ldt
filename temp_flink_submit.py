#!/usr/bin/env python3
import os, sys, time, json

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, WORK_DIR + "/src")

from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types

print(f"[{time.strftime('%H:%M:%S')}] Starting CA-DQStream E2E at {WORK_DIR}")

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
env.get_checkpoint_config().set_checkpoint_interval(60000)

kafka_props = {
    "bootstrap.servers": "ldt-kafka-1:9092",
    "group.id": "cadqstream-e2e",
    "auto.offset.reset": "earliest",
}
kafka_source = FlinkKafkaConsumer(
    topics="taxi-nyc-raw",
    deserialization_schema=SimpleStringSchema(),
    properties=kafka_props
)
stream = env.add_source(kafka_source)

class ParseJson(MapFunction):
    _cnt = 0
    def map(self, v):
        ParseJson._cnt += 1
        if ParseJson._cnt % 10000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Kafka recv: {ParseJson._cnt}")
        try:
            return json.loads(v)
        except:
            return None

stream = stream.map(ParseJson(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)

import importlib
kg = importlib.import_module("operators.key_generator")
gen_id = getattr(kg, "generate_trip_id")
dedup_mod = importlib.import_module("operators.deduplicator")
DedupCls = getattr(dedup_mod, "DeduplicatorFunction")
schema_mod = importlib.import_module("operators.schema_validator")
SchemaCls = getattr(schema_mod, "SchemaValidator")
canary_mod = importlib.import_module("operators.canary_rules")
CanaryCls = getattr(canary_mod, "CanaryRulesValidator")

class AddTripId(MapFunction):
    def map(self, r):
        if r:
            r["trip_id"] = gen_id(r)
        return r

stream = stream.map(AddTripId(), output_type=Types.PICKLED_BYTE_ARRAY())

deduped = (stream
    .key_by(lambda x: x.get("trip_id", ""), key_type=Types.STRING())
    .map(DedupCls(), output_type=Types.PICKLED_BYTE_ARRAY())
    .filter(lambda x: x is not None))

validator = SchemaCls()
valid = deduped.filter(validator)
canary = valid.map(CanaryCls(), output_type=Types.PICKLED_BYTE_ARRAY())

try:
    vec_mod = importlib.import_module("features.vectorizer")
    VecCls = getattr(vec_mod, "FeatureVectorizer")
    ifs_mod = importlib.import_module("operators.if_scoring_operator")
    gck = getattr(ifs_mod, "get_context_key")
    import pickle
    _m = pickle.load(open(WORK_DIR + "/models/iforest_model.pkl", "rb"))
    _s = pickle.load(open(WORK_DIR + "/models/scaler.pkl", "rb"))
    _t = json.load(open(WORK_DIR + "/models/context_thresholds.json"))
    _v = VecCls()
    print(f"[{time.strftime('%H:%M:%S')}] [ML] Loaded: {_m.n_estimators} trees")

    class MLScoring(MapFunction):
        def map(self, r):
            if not r:
                return None
            try:
                f = _v.transform(r)
                fs = _s.transform([f])[0]
                raw = _m.score_samples(fs.reshape(1, -1))[0]
                score = -raw
                ctx = gck(r)
                thr = _t.get("thresholds", {}).get(ctx, _t.get("global_threshold", 0.5))
                return dict(**r, anomaly_score=float(score), threshold=float(thr),
                       is_anomaly=bool(score > thr), context_key=ctx)
            except Exception as e:
                return dict(**r, anomaly_score=0.5, is_anomaly=False, context_key="err")

    ml = valid.map(MLScoring(), output_type=Types.PICKLED_BYTE_ARRAY())
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ML] Skipped: {e}")
    import random
    class MockScoring(MapFunction):
        def map(self, r):
            if not r:
                return None
            return dict(**r, anomaly_score=random.uniform(0.2, 0.8),
                   is_anomaly=False, context_key="mock")
    ml = valid.map(MockScoring(), output_type=Types.PICKLED_BYTE_ARRAY())

valid.filter(lambda x: x).print()
canary.filter(lambda x: x).print()
ml.filter(lambda x: x and x.get("is_anomaly")).print()

print(f"[{time.strftime('%H:%M:%S')}] Submitting job (parallelism={env.get_parallelism()})")
sys.stdout.flush()

result = env.execute("CA-DQStream E2E Test")
print(f"[{time.strftime('%H:%M:%S')}] Job finished: {result}")
