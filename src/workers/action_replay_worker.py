#!/usr/bin/env python3
"""
Action Replay Worker - Retry Failed IEC Actions.
Task 4.6-4.10: Exponential backoff, DLQ, resilient action execution

Consumes from: iec-action-replay topic
Retry strategy: Exponential backoff (2^n seconds)
Max retries: 10
DLQ: Actions that fail 10 times go to Dead Letter Queue

Usage:
  python src/workers/action_replay_worker.py
  docker run cadqstream-action-replay-worker
"""

import argparse
import hashlib
import hmac
import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from kafka import KafkaConsumer, KafkaProducer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


LOGGER = None  # Initialized in main()


def _setup_logger(name: str):
    """Configure structured logger."""
    import logging
    global LOGGER
    LOGGER = logging.getLogger(name)
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        LOGGER.addHandler(handler)
    return LOGGER


def _verify_retrain_signal(data: bytes, hmac_hex: str) -> bool:
    """Verify HMAC-SHA256 signature of a retrain signal.

    FIX CRITICAL #1: Without HMAC verification, a malicious Kafka producer
    could inject fake retrain signals and trigger unauthorized model retraining.
    """
    key = os.environ.get('MEMSTREAM_MODEL_SIGNING_KEY', '')
    if not key:
        LOGGER.warning(
            "MEMSTREAM_MODEL_SIGNING_KEY not set — skipping HMAC verification"
        )
        return True  # Fail open for dev, but log warning

    expected = hmac.new(key.encode(), data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, hmac_hex)


class ActionReplayWorker:
    """Retry failed IEC actions with exponential backoff.

    Architecture:
    - Consumes from iec-action-replay topic
    - Verifies HMAC signatures on incoming signals
    - Idempotency check via processed signal ID set
    - Retries with 2^n exponential backoff + jitter
    - Circuit breaker to fast-fail when ML service is down
    - Dead Letter Queue after max retries exceeded
    - Executes actions via FastAPI ML service
    """

    MAX_TOTAL_RETRY_TIME = 300  # 5 minutes max total retry time

    def __init__(
        self,
        kafka_bootstrap: str = 'localhost:9092',
        ml_service_url: str = 'http://localhost:8000',
        max_retries: int = 10,
        backoff_base: int = 2
    ):
        """Initialize Action Replay Worker.

        Args:
            kafka_bootstrap: Kafka bootstrap servers
            ml_service_url: FastAPI ML service URL
            max_retries: Maximum retry attempts
            backoff_base: Base for exponential backoff (seconds)
        """
        self.kafka_bootstrap = kafka_bootstrap
        self.ml_service_url = ml_service_url
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # FIX CRITICAL #2: Idempotency check
        self._processed_signals: set = set()
        self._processed_max_size = 10000

        # FIX CRITICAL #3: Circuit breaker state
        self._http_failures = 0
        self._http_failure_threshold = 3  # Trip after 3 consecutive failures
        self._http_cooldown_until = 0.0  # Timestamp when cooldown ends

        self._running = True

        # Kafka consumer
        self.consumer = KafkaConsumer(
            'iec-action-replay',
            bootstrap_servers=kafka_bootstrap,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='action-replay-worker',
            auto_offset_reset='earliest'
        )

        # Kafka producer (for retries and DLQ)
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

        # Statistics
        self.stats = {
            'processed': 0,
            'retried': 0,
            'succeeded': 0,
            'dlq': 0,
            'hmac_failures': 0,
            'duplicate_skips': 0,
            'circuit_trips': 0
        }

        self._setup_signal_handlers()

        print("="*60)
        print("Action Replay Worker Initialized")
        print("="*60)
        print(f"Kafka: {kafka_bootstrap}")
        print(f"ML Service: {ml_service_url}")
        print(f"Max Retries: {max_retries}")
        print(f"Backoff Base: {backoff_base}^n seconds")
        print("="*60)

    def _setup_signal_handlers(self):
        """FIX CRITICAL #5: Graceful shutdown on SIGTERM/SIGINT."""
        def _shutdown_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            LOGGER.info(f"Received signal {signum} ({sig_name}), shutting down gracefully...")
            self._running = False
            try:
                self.consumer.commit()
            except Exception as e:
                LOGGER.debug(f"Commit on shutdown: {e}")
            self._print_stats()
            LOGGER.info(f"Shutdown complete. Processed {self.stats['processed']} signals.")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)

    def _process_signal(self, raw_data: dict) -> bool:
        """Verify and process a retrain signal.

        FIX CRITICAL #1: HMAC verification before processing.
        FIX CRITICAL #2: Idempotency check via processed signal set.

        Returns:
            True if signal was processed successfully, False otherwise.
        """
        # FIX CRITICAL #2: Idempotency check
        signal_id = raw_data.get('signal_id', raw_data.get('timestamp', ''))
        if not signal_id:
            signal_id = raw_data.get('action_id', '')

        if signal_id in self._processed_signals:
            LOGGER.info(f"Skipping duplicate signal: {signal_id}")
            self.stats['duplicate_skips'] += 1
            return True

        # FIX CRITICAL #1: HMAC verification
        signal_hmac = raw_data.get('hmac', '')
        signal_body = {k: v for k, v in raw_data.items() if k != 'hmac'}
        signal_bytes = json.dumps(signal_body, sort_keys=True).encode()

        if not _verify_retrain_signal(signal_bytes, signal_hmac):
            LOGGER.error(
                f"HMAC verification failed for signal: {signal_id}"
            )
            self.stats['hmac_failures'] = self.stats.get('hmac_failures', 0) + 1
            return False

        # Mark as processed before executing
        self._processed_signals.add(signal_id)
        if len(self._processed_signals) > self._processed_max_size:
            old_size = len(self._processed_signals)
            # Evict oldest half
            self._processed_signals = set(list(self._processed_signals)[old_size // 2:])

        return True

    def _call_ml_service(self, endpoint: str, payload: dict) -> bool:
        """Call ML service with circuit breaker protection.

        FIX CRITICAL #3: Fast-fail when ML service is clearly down.
        FIX CRITICAL #4: Jitter to prevent thundering herd.

        Returns:
            True if call succeeded, False otherwise.
        """
        # FIX CRITICAL #3: Circuit breaker check
        now = time.time()
        if now < self._http_cooldown_until:
            LOGGER.warning(
                f"HTTP circuit open, skipping until {self._http_cooldown_until:.0f}"
            )
            return False

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=30
            )
            if response.status_code in (200, 201):
                self._http_failures = 0
                return True
            else:
                self._http_failures += 1
                LOGGER.warning(
                    f"ML service returned {response.status_code}: {response.text[:100]}"
                )
        except requests.RequestException as e:
            self._http_failures += 1
            LOGGER.error(f"ML service unavailable: {e}")

        # Trip circuit breaker after threshold failures
        if self._http_failures >= self._http_failure_threshold:
            self._http_cooldown_until = now + 300  # 5 minute cooldown
            self.stats['circuit_trips'] = self.stats.get('circuit_trips', 0) + 1
            LOGGER.error(
                f"HTTP circuit breaker tripped — backing off for 5 minutes "
                f"(consecutive failures: {self._http_failures})"
            )

        return False

    def run(self):
        """Main worker loop."""
        print("\n[Worker] Starting to consume from iec-action-replay...")

        try:
            for message in self.consumer:
                if not self._running:
                    break
                action = message.value
                self.process_action(action)

        except KeyboardInterrupt:
            print("\n[Worker] Shutting down...")
            self._print_stats()

        finally:
            self.consumer.close()
            self.producer.close()

    def process_action(self, action: dict):
        """Process action with retry logic.

        FIX CRITICAL #1: HMAC verification.
        FIX CRITICAL #4: Jitter on backoff.
        FIX CRITICAL #6: Max total retry time cap.

        Args:
            action: Action dict with strategy, retry_count, etc.
        """
        self.stats['processed'] += 1

        # FIX CRITICAL #1: Verify HMAC before processing
        if not self._process_signal(action):
            LOGGER.error(f"Signal verification failed, skipping action")
            return

        retry_count = action.get('retry_count', 0)

        # Check if exceeded max retries
        if retry_count >= self.max_retries:
            LOGGER.warning(
                f"[DLQ] Action exceeded {self.max_retries} retries — "
                f"strategy={action.get('strategy')}, "
                f"neighborhood={action.get('neighborhood')}"
            )
            self._send_to_dlq(action)
            return

        # FIX CRITICAL #4: Exponential backoff with jitter
        if retry_count > 0:
            base_wait = self.backoff_base ** retry_count
            # Add jitter: random.uniform(0.5, 1.5) prevents thundering herd
            jitter = random.uniform(0.5, 1.5)
            backoff_sec = jitter * base_wait
            LOGGER.info(
                f"[Retry {retry_count}/{self.max_retries}] "
                f"Waiting {backoff_sec:.1f}s (jitter={jitter:.2f}) before retry..."
            )
            time.sleep(backoff_sec)

        # Execute action with fast-fail
        success = self._execute_action(action)

        if success:
            LOGGER.info(f"Action succeeded")
            self.stats['succeeded'] += 1
        else:
            # FIX CRITICAL #6: Check max total retry time before scheduling retry
            total_time = sum(
                (self.backoff_base ** i) * 1.0
                for i in range(retry_count + 1)
            )
            if total_time >= self.MAX_TOTAL_RETRY_TIME:
                LOGGER.error(
                    f"Max total retry time ({self.MAX_TOTAL_RETRY_TIME}s) exceeded, "
                    f"sending to DLQ"
                )
                self._send_to_dlq(action)
                return

            LOGGER.info(f"Action failed, scheduling retry...")
            action['retry_count'] = retry_count + 1
            action['last_attempt'] = datetime.utcnow().isoformat()

            self.producer.send('iec-action-replay', action)
            self.stats['retried'] += 1

        # Log stats periodically
        if self.stats['processed'] % 10 == 0:
            self._print_stats()

    def _execute_action(self, action: dict) -> bool:
        """Execute action via FastAPI ML service.

        Args:
            action: Action dict with strategy and parameters

        Returns:
            True if successful, False otherwise
        """
        strategy = action.get('iec_strategy') or action.get('strategy')

        if not strategy:
            LOGGER.warning(f"No strategy in action")
            return False

        # If strategy is do_nothing, acknowledge and skip (already handled)
        if strategy == 'do_nothing' or strategy == 'none':
            LOGGER.info(f"Strategy '{strategy}' — already handled, skipping")
            return True

        # Construct endpoint URL
        endpoint = f"{self.ml_service_url}/api/strategy/{strategy}"

        # Prepare request payload
        payload = {
            'strategy': strategy,
            'meta_metrics': action.get('meta_metrics', {}),
            'neighborhood': action.get('neighborhood', 'unknown')
        }

        # FIX CRITICAL #3: Use circuit-breaker-wrapped call
        return self._call_ml_service(endpoint, payload)

    def _send_to_dlq(self, action: dict):
        """Send action to Dead Letter Queue.

        Args:
            action: Failed action
        """
        # Add DLQ metadata
        action['dlq_timestamp'] = datetime.utcnow().isoformat()
        action['final_retry_count'] = action.get('retry_count', 0)

        # Send to DLQ topic
        self.producer.send('iec-action-dlq', action)

        self.stats['dlq'] += 1

        LOGGER.info(f"Sent to DLQ: strategy={action.get('strategy')}")

    def _print_stats(self):
        """Print worker statistics."""
        print(f"\n{'='*60}")
        print(f"Action Replay Worker Stats")
        print(f"{'='*60}")
        print(f"Processed:        {self.stats['processed']}")
        print(f"Succeeded:       {self.stats['succeeded']}")
        print(f"Retried:         {self.stats['retried']}")
        print(f"DLQ:             {self.stats['dlq']}")
        print(f"HMAC Failures:   {self.stats.get('hmac_failures', 0)}")
        print(f"Duplicate Skips: {self.stats.get('duplicate_skips', 0)}")
        print(f"Circuit Trips:   {self.stats.get('circuit_trips', 0)}")
        print(f"{'='*60}\n")


def main():
    _setup_logger("action_replay_worker")

    parser = argparse.ArgumentParser(description='Action Replay Worker')
    parser.add_argument(
        '--kafka-bootstrap',
        type=str,
        default='localhost:9092',
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--ml-service-url',
        type=str,
        default='http://localhost:8000',
        help='FastAPI ML service URL'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=10,
        help='Maximum retry attempts'
    )
    parser.add_argument(
        '--backoff-base',
        type=int,
        default=2,
        help='Exponential backoff base (seconds)'
    )

    args = parser.parse_args()

    # Create and run worker
    worker = ActionReplayWorker(
        kafka_bootstrap=args.kafka_bootstrap,
        ml_service_url=args.ml_service_url,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base
    )

    worker.run()


if __name__ == '__main__':
    main()
