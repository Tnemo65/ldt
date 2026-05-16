"""
IEC (Intelligent Evolution Controller) Operator - Phase 2D Migration + Phase 5A REC-8.

Task: Multi-strategy adaptation with METER + ADWIN-U

Phase 2D Changes from CA-DQStream version:
1. Uses new IECController from src/iec package with HMAC security
2. Monitoring-only mode by default (no beta writes)
3. After ContextBeta verified: enable beta writes from MinIO
4. Uses 10-neighborhood ADWIN instances from Phase 1D
5. Prometheus alerts for HMAC failures

Phase 5A REC-8: operator_id in all IEC payloads for audit trail.

Flow:
1. Meta-metrics arrive from MetaAggregator
2. ADWIN-U detects drifts per neighborhood×metric
3. METER predicts optimal strategy from metrics
4. IEC executes strategy (monitoring only in Phase 2D)

Usage:
    # Initial migration: monitoring only
    iec_stream = meta_window_stream.map(IECOperator())
    
    # After ContextBeta verified: enable beta writes
    iec_stream = meta_window_stream.map(
        IECOperator(enable_beta_writes=True, minio_client=minio_client)
    )

Spec: Phase 2D migration plan
"""

from pyflink.datastream import MapFunction
import os
import pickle
import socket
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Phase 2D: Import from new IEC package
from src.iec import (
    IECController,
    IECConfig,
    CircuitBreaker,
    CircuitState,
)


def _get_default_operator_id() -> str:
    """Generate default operator_id from hostname + PID + random suffix (REC-8)."""
    hostname = socket.gethostname()
    pid = os.getpid()
    short_uuid = uuid.uuid4().hex[:8]
    return f"iec-{hostname}-{pid}-{short_uuid}"


class IECOperator(MapFunction):
    """
    Intelligent Evolution Controller for adaptive drift handling.
    
    Phase 2D version with HMAC security and monitoring-only mode.

    Combines ADWIN-U drift detection with METER strategy selection.
    Executes multi-strategy adaptation based on drift signals and meta-metrics.

    Strategies:
    1. do_nothing - No drift detected, continue normal operation
    2. adjust_threshold - Minor drift, adjust anomaly thresholds (Phase 2D: monitoring only)
    3. memory_reset - Severe drift, reset memory (Phase 2D: monitoring only)

    Args:
        meter_model_path: Path to trained METER hypernetwork
        meter_scaler_path: Path to METER feature scaler
        adwin_delta_config: ADWIN sensitivity configuration
        enable_beta_writes: Enable beta writes to MinIO (default: False for Phase 2D)
        minio_client: MinIO client for beta writes (required if enable_beta_writes=True)
    """

    def __init__(
        self,
        meter_model_path: str = 'models/meter_hypernetwork.pkl',
        meter_scaler_path: str = 'models/meter_scaler.pkl',
        adwin_delta_config: dict = None,
        enable_beta_writes: bool = False,  # Phase 2D: disabled by default
        minio_client: object = None,
        operator_id: str = None,  # REC-8: audit trail identifier
    ):
        """Initialize IEC operator."""
        self.meter_model_path = meter_model_path
        self.meter_scaler_path = meter_scaler_path
        self.adwin_delta_config = adwin_delta_config
        self.enable_beta_writes = enable_beta_writes  # Phase 2D: disabled by default
        self.minio_client = minio_client
        # REC-8: operator_id for audit trail
        self.operator_id = operator_id or _get_default_operator_id()

        # Phase 2D: Use new IECController
        self.iec_controller: Optional[IECController] = None
        self.iec_config: Optional[IECConfig] = None

        self.stats = {
            'windows_processed': 0,
            'drifts_detected': 0,
            'strategies_executed': {
                'do_nothing': 0,
                'adjust_threshold': 0,
                'memory_reset': 0,
            },
            'hmac_failures': 0,
            'retrain_triggers': {
                'Trigger_A_ADWIN_Drift': 0,
                'Trigger_B_AnomalyRate': 0,
                'Trigger_C_kNNDistance': 0,
            },
        }

    def open(self, runtime_context):
        """
        Initialize IEC controller.

        Called once per task slot when Flink job starts.
        """
        # Phase 2D: Create new IECConfig and IECController
        self.iec_config = IECConfig(
            meter_model_path=self.meter_model_path,
            meter_scaler_path=self.meter_scaler_path,
            enable_meter=True,
        )

        self.iec_controller = IECController(
            config=self.iec_config,
            minio_client=self.minio_client if self.enable_beta_writes else None,
        )

        # Try to load METER model
        if self.iec_controller._load_meter():
            print(f"[IEC] METER model loaded from {self.meter_model_path}")
        else:
            print(f"[IEC] WARNING: METER model not found at {self.meter_model_path}")
            print(f"[IEC] IEC will operate in fallback mode (severity-based strategies)")

        # Log Phase 2D configuration
        print(f"[IEC] Phase 2D IEC Operator initialized")
        print(f"[IEC]   operator_id: {self.operator_id}")  # REC-8
        print(f"[IEC]   Monitoring mode: {'ENABLED' if self.enable_beta_writes else 'MONITORING ONLY'}")
        print(f"[IEC]   Beta writes: {'ENABLED' if self.enable_beta_writes else 'DISABLED'}")
        print(f"[IEC]   HMAC security: ENABLED")

    def map(self, meta_metrics):
        """
        Process meta-metrics window and execute IEC loop.

        Args:
            meta_metrics: Meta-metrics from MetaAggregator (1-minute window)

        Returns:
            IEC decision with strategy and drift status
        """
        if meta_metrics is None:
            return None

        self.stats['windows_processed'] += 1

        # Step 1: Update IEC controller with meta-metrics
        # The meta_metrics format is expected to be a dict keyed by neighborhood:
        # {
        #     'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
        #     'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
        # }
        # Or if it's a flat dict with neighborhood_id field, convert it
        if 'neighborhood_id' in meta_metrics:
            # Flat format - convert to per-neighborhood format
            nb = meta_metrics.get('neighborhood_id', 'unknown')
            meta_metrics = {nb: meta_metrics}

        try:
            decision = self.iec_controller.update(meta_metrics)
        except EnvironmentError as e:
            # HMAC key missing - this should not happen in monitoring mode
            # but log it for visibility
            print(f"[IEC] ERROR: {e}")
            self.stats['hmac_failures'] += 1
            return {
                **meta_metrics,
                'iec_strategy': 'do_nothing',
                'iec_confidence': 0.0,
                'iec_error': str(e),
                'iec_timestamp': datetime.utcnow().isoformat(),
            }

        # Extract decision fields
        strategy = decision.get('strategy', 'do_nothing')
        confidence = decision.get('confidence', 0.0)
        severity = decision.get('severity', 'none')
        drifts = decision.get('drift_events', [])
        retrain_triggers = decision.get('retrain_triggers', {})

        # Update statistics
        self.stats['drifts_detected'] += len(drifts)
        self.stats['strategies_executed'][strategy] = \
            self.stats['strategies_executed'].get(strategy, 0) + 1

        # Track retrain triggers
        for trigger_name in retrain_triggers:
            if trigger_name in self.stats['retrain_triggers']:
                self.stats['retrain_triggers'][trigger_name] += 1

        # Step 2: Execute strategy (if beta writes enabled)
        action_result = {}
        if self.enable_beta_writes:
            try:
                action_result = self.iec_controller.execute_strategy(decision)
            except EnvironmentError as e:
                print(f"[IEC] HMAC ERROR during strategy execution: {e}")
                self.stats['hmac_failures'] += 1
                action_result = {
                    'status': 'hmac_error',
                    'message': str(e),
                }
        else:
            # Monitoring only - log the decision but don't execute
            action_result = {
                'status': 'monitoring_only',
                'message': f'Would execute {strategy} but beta writes disabled',
                'strategy': strategy,
                'confidence': confidence,
                'severity': severity,
            }

        # Emit IEC metrics to cadqstream-metrics for Prometheus
        self._emit_iec_metrics(meta_metrics, drifts, strategy, confidence, severity)

        # Log retrain triggers if any
        if retrain_triggers:
            for trigger_name, trigger_details in retrain_triggers.items():
                print(f"[IEC] RETRAIN TRIGGER: {trigger_name} - {trigger_details}")
                self._emit_retrain_alert(trigger_name, trigger_details)

        # Construct IEC decision (REC-8: includes operator_id for audit trail)
        iec_decision = {
            **meta_metrics,
            'operator_id': self.operator_id,  # REC-8: audit trail
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
            'phase2d_monitoring': not self.enable_beta_writes,
        }

        # Log periodically
        if self.stats['windows_processed'] % 10 == 0:
            self._log_stats()

        return iec_decision

    def _emit_iec_metrics(self, meta_metrics: dict, drifts: list, strategy: str,
                          confidence: float, severity: str):
        """Emit IEC metrics to cadqstream-metrics for Prometheus scraping.

        Sends these metrics:
          - cadqstream_iec_decisions_total{strategy}  (counter)
          - cadqstream_iec_drift_detected_total{neighborhood}  (counter)
          - cadqstream_iec_confidence{neighborhood}  (gauge)
          - cadqstream_meta_anomaly_rate{neighborhood}  (gauge)

        Args:
            meta_metrics: Current window meta-metrics
            drifts: List of detected drifts
            strategy: Selected IEC strategy
            confidence: Strategy confidence score
            severity: Drift severity level
        """
        try:
            import urllib.request
            import json

            # Get neighborhood from meta_metrics
            neighborhoods = list(meta_metrics.keys())
            neighborhood = neighborhoods[0] if neighborhoods else 'global'

            # IEC decision counter
            payload_decision = json.dumps({
                'name': 'iec_decisions_total',
                'value': 1,
                'labels': {
                    'layer': 'L4',
                    'strategy': strategy,
                    'neighborhood': neighborhood,
                    'severity': severity,
                    'operator_id': self.operator_id,  # REC-8
                },
                'type': 'counter'
            }).encode('utf-8')
            req = urllib.request.Request(
                'http://cadqstream-metrics:9250/internal/metrics',
                data=payload_decision,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            try:
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                pass

            # Drift detected counter
            if drifts:
                for drift in drifts:
                    nb = drift.get('neighborhood', neighborhood)
                    payload_drift = json.dumps({
                        'name': 'iec_drift_detected_total',
                        'value': 1,
                        'labels': {
                            'layer': 'L4',
                            'neighborhood': nb,
                            'metric': drift.get('metric', 'unknown'),
                        },
                        'type': 'counter'
                    }).encode('utf-8')
                    req2 = urllib.request.Request(
                        'http://cadqstream-metrics:9250/internal/metrics',
                        data=payload_drift,
                        headers={'Content-Type': 'application/json'},
                        method='POST'
                    )
                    try:
                        urllib.request.urlopen(req2, timeout=2)
                    except Exception:
                        pass

            # IEC confidence gauge
            payload_conf = json.dumps({
                'name': 'iec_confidence',
                'value': float(confidence),
                'labels': {'layer': 'L4', 'neighborhood': neighborhood},
                'type': 'gauge'
            }).encode('utf-8')
            req3 = urllib.request.Request(
                'http://cadqstream-metrics:9250/internal/metrics',
                data=payload_conf,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            try:
                urllib.request.urlopen(req3, timeout=2)
            except Exception:
                pass

            # Meta-metric gauges (from first neighborhood)
            if neighborhoods:
                nb_data = meta_metrics.get(neighborhoods[0], {})
                for metric_name in ['anomaly_rate', 'null_rate', 'violation_rate', 'delta_score']:
                    value = nb_data.get(metric_name, 0.0)
                    payload_metric = json.dumps({
                        'name': f'meta_{metric_name}',
                        'value': float(value),
                        'labels': {'layer': 'L3', 'neighborhood': neighborhoods[0]},
                        'type': 'gauge'
                    }).encode('utf-8')
                    req4 = urllib.request.Request(
                        'http://cadqstream-metrics:9250/internal/metrics',
                        data=payload_metric,
                        headers={'Content-Type': 'application/json'},
                        method='POST'
                    )
                    try:
                        urllib.request.urlopen(req4, timeout=2)
                    except Exception:
                        pass

        except Exception:
            pass

    def _emit_retrain_alert(self, trigger_name: str, trigger_details: dict):
        """Emit Prometheus alert for retrain trigger.
        
        Args:
            trigger_name: Name of the trigger (e.g., 'Trigger_A_ADWIN_Drift')
            trigger_details: Details of the trigger
        """
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
            
            req = urllib.request.Request(
                'http://cadqstream-metrics:9250/internal/metrics',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            try:
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                pass
                
        except Exception:
            pass

    def _log_stats(self):
        """Log IEC statistics."""
        print(f"\n[IEC Statistics - Phase 2D]")
        print(f"  operator_id: {self.operator_id}")  # REC-8
        print(f"  Windows processed: {self.stats['windows_processed']}")
        print(f"  Drifts detected: {self.stats['drifts_detected']}")
        print(f"  HMAC failures: {self.stats['hmac_failures']}")
        print(f"  Strategies executed:")
        for strategy, count in self.stats['strategies_executed'].items():
            print(f"    {strategy}: {count}")
        print(f"  Retrain triggers:")
        for trigger, count in self.stats['retrain_triggers'].items():
            print(f"    {trigger}: {count}")
        if self.iec_controller:
            circuit_status = self.iec_controller.get_circuit_status()
            print(f"  Circuit breaker: {circuit_status.get('state', 'unknown')}")
            print(f"    Trip count: {circuit_status.get('trip_count', 0)}")
            print(f"    Consecutive actions: {circuit_status.get('consecutive_actions', 0)}")

    def close(self):
        """Print final statistics on close."""
        print(f"\n{'='*60}")
        print(f"IEC OPERATOR PHASE 2D - FINAL STATISTICS")
        print(f"{'='*60}")
        self._log_stats()
        print(f"{'='*60}")
