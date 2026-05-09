#!/usr/bin/env python3
import os, sys, time, multiprocessing

WORK = "/opt/flink/e2e"
LOG = WORK + "/job.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")
        f.flush()

def run():
    log("Process starting at " + time.strftime("%Y-%m-%d %H:%M:%S"))
    sys.path.insert(0, WORK)
    sys.path.insert(0, WORK + "/src")
    os.environ["PYTHONPATH"] = WORK + ":" + WORK + "/src"

    try:
        log("Importing Flink...")
        from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, MapFunction
        from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
        from pyflink.common.serialization import SimpleStringSchema
        from pyflink.common.typeinfo import Types

        log("Creating environment...")
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_runtime_mode(RuntimeExecutionExecutionMode.STREAMING)
        env.set_parallelism(4)
        env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
        env.get_checkpoint_config().set_checkpoint_interval(60000)

        jm_host = os.environ.get("FLINK_JM_HOST", "ldt-flink-jobmanager")
        log("Setting REST URL to " + jm_host + ":8081")
        env.get_checkpoint_config().get_configuration().set_string("rest.address", jm_host)
        env.get_checkpoint_config().get_configuration().set_integer("rest.port", 8081)
        env.set_rest_url(jm_host, 8081)

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

        class PJ(MapFunction):
            _cnt = 0
            def map(self, v):
                PJ._cnt += 1
                if PJ._cnt % 10000 == 0:
                    log("  Kafka recv: " + str(PJ._cnt))
                try:
                    import json
                    return json.loads(v)
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
            import pickle, json as js
            _m = pickle.load(open(WORK + "/models/iforest_model.pkl", "rb"))
            _s = pickle.load(open(WORK + "/models/scaler.pkl", "rb"))
            _t = js.load(open(WORK + "/models/context_thresholds.json"))
            _v = VecCls()
            log("[ML] Loaded: " + str(_m.n_estimators) + " trees")

            class ML(MapFunction):
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
                    except:
                        return dict(**r, anomaly_score=0.5, is_anomaly=False, context_key="err")

            ml = valid.map(ML(), output_type=Types.PICKLED_BYTE_ARRAY())
        except Exception as e:
            log("[ML] Skipped: " + str(e))
            import random
            class Mock(MapFunction):
                def map(self, r):
                    if not r:
                        return None
                    return dict(**r, anomaly_score=random.uniform(0.2, 0.8),
                           is_anomaly=False, context_key="mock")
            ml = valid.map(Mock(), output_type=Types.PICKLED_BYTE_ARRAY())

        log("Routing outputs...")
        valid.filter(lambda x: x).print()
        canary.filter(lambda x: x).print()
        ml.filter(lambda x: x and x.get("is_anomaly")).print()

        log("Starting env.execute()...")
        log("Parallelism: " + str(env.get_parallelism()))
        sys.stdout.flush()

        result = env.execute("CA-DQStream E2E Test")
        log("Job finished: " + str(result))
    except Exception as e:
        import traceback
        log("ERROR: " + str(e))
        tb = traceback.format_exc()
        log(tb)

    log("Process exiting at " + time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join(timeout=300)
    if p.is_alive():
        log("Process timed out, terminating...")
        p.terminate()
        p.join()
    log("Done")
