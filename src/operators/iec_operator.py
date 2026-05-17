"""
IEC Operator (Flink MapFunction) - Sequential Pipeline Phase 3.

Task: Sequential adaptive drift handling with 2 strategies.

Phase 3 Changes from Phase 2D:
1. NO METER hypernetwork — severity-based strategy selection only
2. NO beta writes — removed from new flow
3. 2 strategies: do_nothing, quick_retrain
4. Retrain signals written to MinIO for action-replay-worker to consume

Flow:
1. Meta-metrics arrive from MetaAggregator (1-minute window per neighborhood)
2. ADWIN-U detects drifts per neighborhood x metric
3. IEC assesses severity and selects strategy
4. execute_strategy() writes retrain signal to MinIO if quick_retrain

Usage:
    iec_stream = meta_window_stream.map(IECOperator())
"""

from pyflink.datastream import MapFunction
import os
import socket
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.iec import (
    IECController,
    IECConfig,
    CircuitBreaker,
    CircuitState,
)


def _get_default_operator_id() -> str:
    """Generate default operator_id from hostname + PID + random suffix."""
    hostname = socket.gethostname()
    pid = os.getpid()
    short_uuid = uuid.uuid4().hex[:8]
    return f"iec-{hostname}-{pid}-{short_uuid}"


def _get_minio_client():
    """Create MinIO client for retrain signal writes."""
    try:
        import boto3
        from botocore.config import Config

        endpoint = os.getenv('MINIO_ENDPOINT', os.getenv('S3_ENDPOINT', 'http://minio:9000'))
        endpoint = endpoint.strip() if endpoint else 'http://minio:9000'
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = 'http://' + endpoint

        cfg = Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'},
            connect_timeout=5.0,
            read_timeout=30.0,
            s3={'addressing_style': 'path'},
        )

        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')),
            config=cfg,
        )
    except Exception as e:
        print(f"[IEC] WARNING: Could not create MinIO client: {e}")
        return None


class IECOperator(MapFunction):
    """
    Intelligent Evolution Controller for adaptive drift handling.

    Phase 3: Sequential pipeline, 2 strategies.

    Strategies:
    1. do_nothing - No drift or minor drift, continue normal operation
    2. quick_retrain - Severe drift, retrain MemStream AE + reset ADWIN thresholds

    Args:
        operator_id: Audit trail identifier (auto-generated if not provided)
    """

    def __init__(
        self,
        operator_id: Optional[str] = None,
    ):
        self.operator_id = operator_id or _get_default_operator_id()
        self.iec_controller: Optional[IECController] = None
        self.iec_config: Optional[IECConfig] = None
        self._minio_client = None

        self.stats = {
            'windows_processed': 0,
            'drifts_detected': 0,
            'strategies_executed': {
                'do_nothing': 0,
                'quick_retrain': 0,
            },
            'retrain_triggers': {
                'Trigger_A_ADWIN_Drift': 0,
                'Trigger_B_AnomalyRate': 0,
                'Trigger_C_kNNDistance': 0,
            },
        }

    def open(self, runtime_context):
        """Initialize IEC controller and MinIO client."""
        self._minio_client = _get_minio_client()

        self.iec_config = IECConfig()

        self.iec_controller = IECController(
            config=self.iec_config,
            minio_client=self._minio_client,
        )

        print(f"[IEC] Phase 3 IEC Operator initialized")
        print(f"[IEC]   operator_id: {self.operator_id}")
        print(f"[IEC]   Strategies: do_nothing, quick_retrain")
        print(f"[IEC]   Retrain triggers: ADWIN drift, AnomalyRate, kNNDistance")
        print(f"[IEC]   MinIO client: {'CONNECTED' if self._minio_client else 'NOT AVAILABLE'}")

    def map(self, meta_metrics):
        """
        Process meta-metrics window and execute IEC loop.

        Args:
            meta_metrics: Meta-metrics from MetaAggregator.
                Can be a flat dict with neighborhood_id field, or a dict keyed
                by neighborhood name.

        Returns:
            IEC decision with strategy and drift status
        """
        if meta_metrics is None:
            return None

        self.stats['windows_processed'] += 1

        # Normalize format: if flat dict, wrap in {neighborhood: metrics}
        if 'neighborhood_id' in meta_metrics:
            nb = meta_metrics.get('neighborhood_id', 'unknown')
            meta_metrics = {nb: meta_metrics}

        try:
            decision = self.iec_controller.update(meta_metrics)
        except Exception as e:
            print(f"[IEC] ERROR in update: {e}")
            return {
                **meta_metrics,
                'iec_strategy': 'do_nothing',
                'iec_confidence': 0.0,
                'iec_error': str(e),
                'iec_timestamp': datetime.utcnow().isoformat(),
            }

        strategy = decision.get('strategy', 'do_nothing')
        confidence = decision.get('confidence', 0.0)
        severity = decision.get('severity', 'none')
        drifts = decision.get('drift_events', [])
        retrain_triggers = decision.get('retrain_triggers', {})

        self.stats['drifts_detected'] += len(drifts)
        self.stats['strategies_executed'][strategy] = \
            self.stats['strategies_executed'].get(strategy, 0) + 1

        for trigger_name in retrain_triggers:
            if trigger_name in self.stats['retrain_triggers']:
                self.stats['retrain_triggers'][trigger_name] += 1

        # Execute strategy
        try:
            action_result = self.iec_controller.execute_strategy(decision)
        except Exception as e:
            print(f"[IEC] ERROR in execute_strategy: {e}")
            action_result = {'status': 'error', 'message': str(e)}

        # Emit metrics
        self._emit_iec_metrics(meta_metrics, drifts, strategy, confidence, severity)

        if retrain_triggers:
            for trigger_name, trigger_details in retrain_triggers.items():
                print(f"[IEC] RETRAIN TRIGGER: {trigger_name} - {trigger_details}")
                self._emit_retrain_alert(trigger_name, trigger_details)

        iec_decision = {
            'operator_id': self.operator_id,
            'drifts_detected': drifts,
            'drift_count': len(drifts),
            'drift_assessment': severity,
            'iec_strategy': strategy,
            'iec_confidence': confidence,
            'iec_severity': severity,
            'action_result': action_result,
            'retrain_triggers': retrain_triggers,
            'circuit_state': decision.get('circuit_state', 'closed'),
            'iec_timestamp': datetime.utcnow().isoformat(),
        }

        # Merge neighborhood data into output
        for nb, nb_metrics in meta_metrics.items():
            iec_decision[f'nb_{nb}_volume'] = nb_metrics.get('volume', 0)
            iec_decision[f'nb_{nb}_anomaly_rate'] = nb_metrics.get('anomaly_rate', 0.0)
            iec_decision[f'nb_{nb}_violation_rate'] = nb_metrics.get('violation_rate', 0.0)
            iec_decision[f'nb_{nb}_delta_score'] = nb_metrics.get('delta_score', 0.0)

        if self.stats['windows_processed'] % 10 == 0:
            self._log_stats()

        return iec_decision

    def _emit_iec_metrics(self, meta_metrics: dict, drifts: list, strategy: str,
                          confidence: float, severity: str):
        """Emit IEC metrics to cadqstream-metrics for Prometheus."""
        try:
            import urllib.request
            import json

            neighborhoods = list(meta_metrics.keys())
            neighborhood = neighborhoods[0] if neighborhoods else 'global'

            # IEC decision counter
            payload = json.dumps({
                'name': 'iec_decisions_total',
                'value': 1,
                'labels': {
                    'layer': 'L4',
                    'strategy': strategy,
                    'neighborhood': neighborhood,
                    'severity': severity,
                    'operator_id': self.operator_id,
                },
                'type': 'counter'
            }).encode('utf-8')
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        'http://cadqstream-metrics:9250/internal/metrics',
                        data=payload,
                        headers={'Content-Type': 'application/json'},
                        method='POST'
                    ),
                    timeout=2
                )
            except Exception:
                pass

            # Drift detected counter
            for drift in drifts:
                nb = drift.get('neighborhood', neighborhood)
                p2 = json.dumps({
                    'name': 'iec_drift_detected_total',
                    'value': 1,
                    'labels': {
                        'layer': 'L4',
                        'neighborhood': nb,
                        'metric': drift.get('metric', 'unknown'),
                    },
                    'type': 'counter'
                }).encode('utf-8')
                try:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            'http://cadqstream-metrics:9250/internal/metrics',
                            data=p2,
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        ),
                        timeout=2
                    )
                except Exception:
                    pass

            # IEC confidence gauge
            p3 = json.dumps({
                'name': 'iec_confidence',
                'value': float(confidence),
                'labels': {'layer': 'L4', 'neighborhood': neighborhood},
                'type': 'gauge'
            }).encode('utf-8')
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        'http://cadqstream-metrics:9250/internal/metrics',
                        data=p3,
                        headers={'Content-Type': 'application/json'},
                        method='POST'
                    ),
                    timeout=2
                )
            except Exception:
                pass

        except Exception:
            pass

    def _emit_retrain_alert(self, trigger_name: str, trigger_details: dict):
        """Emit Prometheus alert for retrain trigger."""
        try:
            import urllib.request
            import json

            neighborhood = trigger_details.get('neighborhood', 'unknown')

            payload = json.dumps({
                'name': 'retrain_trigger_total',
                'value': 1,
                'labels': {
                    'layer': 'L4',
                    'trigger': trigger_name,
                    'neighborhood': neighborhood,
                },
                'type': 'counter'
            }).encode('utf-8')

            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        'http://cadqstream-metrics:9250/internal/metrics',
                        data=payload,
                        headers={'Content-Type': 'application/json'},
                        method='POST'
                    ),
                    timeout=2
                )
            except Exception:
                pass
        except Exception:
            pass

    def _log_stats(self):
        """Log IEC statistics."""
        print(f"\n[IEC Statistics - Phase 3]")
        print(f"  operator_id: {self.operator_id}")
        print(f"  Windows processed: {self.stats['windows_processed']}")
        print(f"  Drifts detected: {self.stats['drifts_detected']}")
        print(f"  Strategies executed:")
        for strategy, count in self.stats['strategies_executed'].items():
            print(f"    {strategy}: {count}")
        print(f"  Retrain triggers:")
        for trigger, count in self.stats['retrain_triggers'].items():
            print(f"    {trigger}: {count}")
        if self.iec_controller:
            circuit = self.iec_controller.get_circuit_status()
            print(f"  Circuit breaker: {circuit.get('state', 'unknown')}")
            print(f"    Trip count: {circuit.get('trip_count', 0)}")
            print(f"    Consecutive actions: {circuit.get('consecutive_actions', 0)}")

    def close(self):
        """Print final statistics on close."""
        print(f"\n{'=' * 60}")
        print(f"IEC OPERATOR PHASE 3 - FINAL STATISTICS")
        print(f"{'=' * 60}")
        self._log_stats()
        print(f"{'=' * 60}")
