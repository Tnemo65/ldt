#!/usr/bin/env python3
"""
MemStream Chaos Engineering Test Suite (Phase 5B).

This module provides failure injection scenarios for testing MemStream's
resilience and graceful degradation capabilities.

Test Scenarios:
1. Checkpoint Corruption - Verify graceful degradation and recovery
2. Redis Loss - Verify IEC degrades silently
3. Model HMAC Failure - Verify hard failure + alert

Usage:
    python -m deployment.chaos.test_checkpoint_corruption
    python -m deployment.chaos.test_redis_loss
    python -m deployment.chaos.test_hmac_failure

Requirements:
    - boto3 (for MinIO manipulation)
    - redis (for Redis manipulation)
    - kafka-python (for Kafka metrics)
    - prometheus-api-client (for Prometheus queries)
"""

import hashlib
import hmac
import io
import json
import logging
import os
import random
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import boto3
from botocore.config import Config

LOGGER = logging.getLogger('memstream-chaos')


class FailureScenario(Enum):
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    REDIS_LOSS = "redis_loss"
    MODEL_HMAC_FAILURE = "model_hmac_failure"


@dataclass
class ChaosTestResult:
    scenario: FailureScenario
    start_time: datetime
    end_time: Optional[datetime] = None
    injection_success: bool = False
    expected_behavior_verified: bool = False
    actual_behavior: str = ""
    recovery_time_seconds: Optional[float] = None
    alerts_triggered: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class MinIOClient:
    """MinIO/S3 client for checkpoint manipulation."""

    def __init__(self):
        self.endpoint = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY', '')
        self.secret_key = os.getenv('MINIO_SECRET_KEY', '')
        self.bucket = os.getenv('MEMSTREAM_CHECKPOINT_BUCKET', 'cadqstream-drift')
        self.prefix = os.getenv('MEMSTREAM_CHECKPOINT_PREFIX', 'checkpoints/memstream/')

        cfg = Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'},
            connect_timeout=5.0,
            read_timeout=30.0,
            s3={'addressing_style': 'path'},
        )
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=cfg,
        )

    def get_checkpoint_key(self) -> str:
        return f"{self.prefix}memstream_memory.pt"

    def get_checkpoint_hmac_key(self) -> str:
        return f"{self.prefix}memstream_memory.pt.hmac"

    def download_checkpoint(self) -> Optional[bytes]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.get_checkpoint_key())
            return response['Body'].read()
        except Exception as e:
            LOGGER.error("Failed to download checkpoint: %s", e)
            return None

    def upload_checkpoint(self, data: bytes, content_type: str = 'application/octet-stream'):
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.get_checkpoint_key(),
            Body=data,
            ContentType=content_type,
        )

    def delete_checkpoint(self):
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self.get_checkpoint_key())
        except Exception as e:
            LOGGER.warning("Failed to delete checkpoint: %s", e)

    def inject_corruption(self, corruption_type: str = 'random_bytes') -> bool:
        """
        Inject corruption into checkpoint file.

        Args:
            corruption_type: 'random_bytes', 'truncate', 'flip_bits', 'invalid_torch'

        Returns:
            True if corruption was successfully injected
        """
        data = self.download_checkpoint()
        if data is None:
            LOGGER.error("No checkpoint to corrupt")
            return False

        if corruption_type == 'random_bytes':
            corrupted = bytes([random.randint(0, 255) for _ in range(min(len(data), 100))])
            corrupted += data[100:]
        elif corruption_type == 'truncate':
            corrupted = data[:len(data) // 2]
        elif corruption_type == 'flip_bits':
            byte_array = bytearray(data)
            for i in range(min(100, len(byte_array))):
                byte_array[i] ^= 0xFF
            corrupted = bytes(byte_array)
        elif corruption_type == 'invalid_torch':
            corrupted = b'This is not a valid torch checkpoint'
        else:
            LOGGER.error("Unknown corruption type: %s", corruption_type)
            return False

        self.upload_checkpoint(corrupted)
        LOGGER.info("Checkpoint corrupted with type: %s", corruption_type)
        return True


class PrometheusMetrics:
    """Prometheus metrics client for verifying alerts."""

    def __init__(self, prometheus_url: str = 'http://prometheus:9090'):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.session = None

    def query(self, query: str) -> List[Dict]:
        import requests
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'success':
                return data.get('data', {}).get('result', [])
            return []
        except Exception as e:
            LOGGER.error("Prometheus query failed: %s", e)
            return []

    def get_alert_state(self, alert_name: str) -> Optional[str]:
        """Get current state of an alert (Pending, Firing, Inactive)."""
        results = self.query(f'alertmanager_alerts{{alertname="{alert_name}"}}')
        if results:
            return results[0].get('labels', {}).get('alertstate', 'unknown')
        return None

    def get_metric_value(self, metric_name: str, label_selectors: Dict = None) -> Optional[float]:
        """Get current value of a metric."""
        query = metric_name
        if label_selectors:
            selectors = ','.join([f'{k}="{v}"' for k, v in label_selectors.items()])
            query = f'{metric_name}{{{selectors}}}'
        results = self.query(query)
        if results:
            value = results[0].get('value', [None, None])
            if len(value) == 2:
                try:
                    return float(value[1])
                except (ValueError, TypeError):
                    pass
        return None


class KafkaClient:
    """Kafka client for checking consumer lag during tests."""

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers

    def get_consumer_lag(self, group: str) -> Optional[int]:
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                group_id=group,
                auto_offset_reset='latest',
                enable_auto_commit=True,
            )
            metrics = consumer.metrics()
            lag = metrics.get('consumer-lag', {}).get('value', 0)
            consumer.close()
            return int(lag) if lag else None
        except Exception as e:
            LOGGER.warning("Failed to get consumer lag: %s", e)
            return None


class MemStreamChaosEngine:
    """
    Orchestrates chaos engineering tests for MemStream.

    Usage:
        engine = MemStreamChaosEngine()
        result = engine.run_scenario(FailureScenario.CHECKPOINT_CORRUPTION)
    """

    def __init__(self):
        self.minio = MinIOClient()
        self.prometheus = PrometheusMetrics()
        self.kafka = KafkaClient()
        self.test_results: List[ChaosTestResult] = []

    def verify_checkpoint_corruption_alert(self, timeout_seconds: int = 60) -> Tuple[bool, List[str]]:
        """
        Verify that checkpoint corruption triggers the expected alert.

        Expected: MemStream_CheckpointCorruption alert fires within timeout.
        """
        alerts = []
        start = time.time()

        while time.time() - start < timeout_seconds:
            corruption_alerts = self.prometheus.query(
                'ALERTS{alertname="MemStream_CheckpointCorruption",alertstate="firing"}'
            )
            if corruption_alerts:
                alerts.append("MemStream_CheckpointCorruption")

            checkpoint_size = self.prometheus.get_metric_value('memstream_checkpoint_size_bytes')
            if checkpoint_size and checkpoint_size == 0:
                alerts.append("checkpoint_size_zero")

            if alerts:
                break
            time.sleep(2)

        return len(alerts) > 0, alerts

    def verify_graceful_degradation(self, timeout_seconds: int = 120) -> Tuple[bool, str]:
        """
        Verify that system continues operating with degraded quality.

        Expected: Anomaly rate may increase but scoring continues.
        """
        start = time.time()
        checkpoint_restore_detected = False

        while time.time() - start < timeout_seconds:
            anomaly_rate = self.prometheus.get_metric_value('memstream_anomaly_rate')
            scoring_rate = self.prometheus.get_metric_value(
                'rate(memstream_records_processed_total[5m])'
            )

            if scoring_rate and scoring_rate > 0:
                if anomaly_rate and anomaly_rate < 0.5:
                    checkpoint_restore_detected = True

            if checkpoint_restore_detected:
                break
            time.sleep(5)

        behavior = "degraded_operation" if checkpoint_restore_detected else "service_impacted"
        return checkpoint_restore_detected, behavior

    def verify_iec_silent_degradation(self, timeout_seconds: int = 60) -> Tuple[bool, str]:
        """
        Verify that Redis loss causes silent degradation (no hard errors).

        Expected: Beta staleness increases but scoring continues.
        """
        start = time.time()
        staleness_increased = False
        errors_remained_low = True

        while time.time() - start < timeout_seconds:
            staleness = self.prometheus.get_metric_value('max(memstream_beta_staleness_seconds)')
            hmac_errors = self.prometheus.get_metric_value(
                'rate(memstream_hmac_verification_failures_total[5m])'
            )

            if staleness and staleness > 100:
                staleness_increased = True

            if hmac_errors and hmac_errors > 0:
                errors_remained_low = False

            if staleness_increased and errors_remained_low:
                break
            time.sleep(2)

        behavior = "silent_degradation" if (staleness_increased and errors_remained_low) else "unexpected_behavior"
        return staleness_increased and errors_remained_low, behavior

    def verify_hmac_hard_failure(self, timeout_seconds: int = 30) -> Tuple[bool, List[str]]:
        """
        Verify that HMAC failure causes hard failure with alert.

        Expected: MemStream_ModelHMACFailure alert fires, scoring may pause.
        """
        alerts = []
        start = time.time()

        while time.time() - start < timeout_seconds:
            hmac_alerts = self.prometheus.query(
                'ALERTS{alertname="MemStream_ModelHMACFailure",alertstate="firing"}'
            )
            security_alerts = self.prometheus.query(
                'ALERTS{alertname="MemStream_HMACVerificationFailures",alertstate="firing"}'
            )

            if hmac_alerts:
                alerts.append("MemStream_ModelHMACFailure")
            if security_alerts:
                alerts.append("MemStream_HMACVerificationFailures")

            if alerts:
                break
            time.sleep(2)

        return len(alerts) > 0, alerts

    def run_checkpoint_corruption_test(
        self,
        corruption_type: str = 'random_bytes',
        verify_duration_seconds: int = 120
    ) -> ChaosTestResult:
        """Run checkpoint corruption chaos test."""
        result = ChaosTestResult(
            scenario=FailureScenario.CHECKPOINT_CORRUPTION,
            start_time=datetime.now()
        )

        try:
            LOGGER.info("=== Starting Checkpoint Corruption Test ===")

            # Step 1: Inject corruption
            injection_success = self.minio.inject_corruption(corruption_type)
            result.injection_success = injection_success

            if not injection_success:
                result.error_message = "Failed to inject corruption"
                return result

            LOGGER.info("Corruption injected successfully")

            # Step 2: Wait for detection
            time.sleep(5)

            # Step 3: Verify alert triggers
            alerts_triggered, alert_names = self.verify_checkpoint_corruption_alert(timeout=30)
            result.alerts_triggered = alert_names

            if not alerts_triggered:
                LOGGER.warning("Expected alert did not fire, checking graceful degradation anyway")

            # Step 4: Verify graceful degradation
            time.sleep(10)
            degradation_ok, behavior = self.verify_graceful_degradation(
                timeout=verify_duration_seconds
            )
            result.expected_behavior_verified = degradation_ok
            result.actual_behavior = behavior

            # Step 5: Wait for recovery
            recovery_start = time.time()
            while time.time() - recovery_start < 60:
                checkpoint_size = self.prometheus.get_metric_value('memstream_checkpoint_size_bytes')
                if checkpoint_size and checkpoint_size > 0:
                    result.recovery_time_seconds = time.time() - recovery_start
                    break
                time.sleep(5)

            result.end_time = datetime.now()
            LOGGER.info("=== Checkpoint Corruption Test Complete ===")
            LOGGER.info("  Injection: %s", result.injection_success)
            LOGGER.info("  Alerts: %s", result.alerts_triggered)
            LOGGER.info("  Degradation OK: %s", result.expected_behavior_verified)
            LOGGER.info("  Recovery Time: %s", result.recovery_time_seconds)

        except Exception as e:
            result.error_message = str(e)
            LOGGER.error("Test failed with error: %s", e)

        self.test_results.append(result)
        return result

    def run_redis_loss_test(
        self,
        redis_host: str = 'redis',
        redis_port: int = 6379,
        kill_duration_seconds: int = 60,
        verify_duration_seconds: int = 60
    ) -> ChaosTestResult:
        """Run Redis loss chaos test."""
        result = ChaosTestResult(
            scenario=FailureScenario.REDIS_LOSS,
            start_time=datetime.now()
        )

        try:
            LOGGER.info("=== Starting Redis Loss Test ===")

            # Step 1: Kill Redis connection (simulate by setting a flag)
            import redis
            try:
                r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=5)
                r.client_setname('chaos-test-disconnect')
                r.shutdown(nosave=True)
            except redis.ConnectionError:
                LOGGER.info("Redis connection already lost or shutdown command not permitted")
            except Exception as e:
                LOGGER.warning("Could not shutdown Redis: %s (continuing with test)", e)

            # Step 2: Wait for detection
            time.sleep(10)

            # Step 3: Verify silent degradation
            degradation_ok, behavior = self.verify_iec_silent_degradation(
                timeout=verify_duration_seconds
            )
            result.expected_behavior_verified = degradation_ok
            result.actual_behavior = behavior

            # Step 4: Check for alert (should NOT fire for silent degradation)
            redis_alerts = self.prometheus.query(
                'ALERTS{alertname="MemStream_RedisConnectionFailure",alertstate="firing"}'
            )
            if redis_alerts:
                result.alerts_triggered.append("MemStream_RedisConnectionFailure")

            result.end_time = datetime.now()
            LOGGER.info("=== Redis Loss Test Complete ===")
            LOGGER.info("  Silent Degradation OK: %s", result.expected_behavior_verified)
            LOGGER.info("  Behavior: %s", result.actual_behavior)
            LOGGER.info("  Alerts: %s", result.alerts_triggered)

        except Exception as e:
            result.error_message = str(e)
            LOGGER.error("Test failed with error: %s", e)

        self.test_results.append(result)
        return result

    def run_hmac_failure_test(
        self,
        signing_key: str = None,
        verify_duration_seconds: int = 30
    ) -> ChaosTestResult:
        """Run model HMAC failure chaos test."""
        result = ChaosTestResult(
            scenario=FailureScenario.MODEL_HMAC_FAILURE,
            start_time=datetime.now()
        )

        try:
            LOGGER.info("=== Starting Model HMAC Failure Test ===")

            # Step 1: Get current checkpoint
            checkpoint_data = self.minio.download_checkpoint()
            if checkpoint_data is None:
                result.error_message = "No checkpoint to test"
                return result

            # Step 2: Corrupt the HMAC file with wrong signature
            wrong_hmac = 'deadbeef' * 8  # Obviously wrong HMAC
            try:
                self.minio.client.put_object(
                    Bucket=self.minio.bucket,
                    Key=self.minio.get_checkpoint_hmac_key(),
                    Body=wrong_hmac.encode(),
                    ContentType='text/plain',
                )
                result.injection_success = True
            except Exception as e:
                LOGGER.warning("Could not write HMAC file: %s", e)
                result.injection_success = False

            LOGGER.info("HMAC file corrupted with wrong signature")

            # Step 3: Wait for detection
            time.sleep(5)

            # Step 4: Verify hard failure and alert
            hard_failure_ok, alert_names = self.verify_hmac_hard_failure(
                timeout=verify_duration_seconds
            )
            result.alerts_triggered = alert_names
            result.expected_behavior_verified = hard_failure_ok
            result.actual_behavior = "hard_failure_with_alert" if hard_failure_ok else "unexpected_behavior"

            result.end_time = datetime.now()
            LOGGER.info("=== HMAC Failure Test Complete ===")
            LOGGER.info("  Injection: %s", result.injection_success)
            LOGGER.info("  Hard Failure Verified: %s", result.expected_behavior_verified)
            LOGGER.info("  Alerts: %s", result.alerts_triggered)

        except Exception as e:
            result.error_message = str(e)
            LOGGER.error("Test failed with error: %s", e)

        self.test_results.append(result)
        return result

    def run_scenario(
        self,
        scenario: FailureScenario,
        **kwargs
    ) -> ChaosTestResult:
        """Run a specific chaos test scenario."""
        if scenario == FailureScenario.CHECKPOINT_CORRUPTION:
            return self.run_checkpoint_corruption_test(**kwargs)
        elif scenario == FailureScenario.REDIS_LOSS:
            return self.run_redis_loss_test(**kwargs)
        elif scenario == FailureScenario.MODEL_HMAC_FAILURE:
            return self.run_hmac_failure_test(**kwargs)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def run_all_scenarios(self) -> List[ChaosTestResult]:
        """Run all chaos test scenarios in sequence."""
        results = []

        for scenario in FailureScenario:
            LOGGER.info("Running scenario: %s", scenario.value)
            result = self.run_scenario(scenario)
            results.append(result)

            # Cool down between tests
            time.sleep(30)

        return results

    def generate_report(self) -> str:
        """Generate a test report."""
        report_lines = [
            "=" * 70,
            "MemStream Chaos Engineering Test Report",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        for result in self.test_results:
            report_lines.extend([
                "-" * 70,
                f"Scenario: {result.scenario.value}",
                f"Duration: {result.duration_seconds:.1f}s" if result.duration_seconds else "N/A",
                f"Injection Success: {result.injection_success}",
                f"Expected Behavior Verified: {result.expected_behavior_verified}",
                f"Actual Behavior: {result.actual_behavior}",
                f"Alerts Triggered: {', '.join(result.alerts_triggered) if result.alerts_triggered else 'None'}",
                f"Recovery Time: {result.recovery_time_seconds:.1f}s" if result.recovery_time_seconds else "N/A",
                f"Error: {result.error_message}" if result.error_message else "",
                "",
            ])

        summary = {
            'total': len(self.test_results),
            'passed': sum(1 for r in self.test_results if r.expected_behavior_verified),
            'failed': sum(1 for r in self.test_results if not r.expected_behavior_verified),
        }
        report_lines.extend([
            "=" * 70,
            "Summary",
            "=" * 70,
            f"Total Tests: {summary['total']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Pass Rate: {summary['passed']/summary['total']*100:.1f}%",
        ])

        return "\n".join(report_lines)


def main():
    """Main entry point for running chaos tests."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='MemStream Chaos Engineering Tests')
    parser.add_argument(
        '--scenario',
        choices=['all', 'checkpoint_corruption', 'redis_loss', 'hmac_failure'],
        default='all',
        help='Which scenario to run'
    )
    parser.add_argument(
        '--corruption-type',
        choices=['random_bytes', 'truncate', 'flip_bits', 'invalid_torch'],
        default='random_bytes',
        help='Type of corruption to inject'
    )
    parser.add_argument(
        '--output',
        help='Output file for test report'
    )

    args = parser.parse_args()

    engine = MemStreamChaosEngine()

    if args.scenario == 'all':
        results = engine.run_all_scenarios()
    else:
        scenario = FailureScenario(args.scenario)
        results = [engine.run_scenario(scenario, corruption_type=args.corruption_type)]

    report = engine.generate_report()
    print(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")

    # Exit with non-zero if any tests failed
    failed = sum(1 for r in results if not r.expected_behavior_verified)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
