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
import sys
from pathlib import Path
import json
import time
import requests
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ActionReplayWorker:
    """Retry failed IEC actions with exponential backoff.

    Architecture:
    - Consumes from iec-action-replay topic
    - Retries with 2^n exponential backoff (1s, 2s, 4s, 8s, ...)
    - Dead Letter Queue after 10 failures
    - Executes actions via FastAPI ML service
    """

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
            'dlq': 0
        }

        print("="*60)
        print("Action Replay Worker Initialized")
        print("="*60)
        print(f"Kafka: {kafka_bootstrap}")
        print(f"ML Service: {ml_service_url}")
        print(f"Max Retries: {max_retries}")
        print(f"Backoff Base: {backoff_base}^n seconds")
        print("="*60)

    def run(self):
        """Main worker loop."""
        print("\n[Worker] Starting to consume from iec-action-replay...")

        try:
            for message in self.consumer:
                action = message.value
                self.process_action(action)

        except KeyboardInterrupt:
            print("\n[Worker] Shutting down...")
            self.print_stats()

        finally:
            self.consumer.close()
            self.producer.close()

    def process_action(self, action: dict):
        """Process action with retry logic.

        Args:
            action: Action dict with strategy, retry_count, etc.
        """
        self.stats['processed'] += 1

        retry_count = action.get('retry_count', 0)

        # Check if exceeded max retries
        if retry_count >= self.max_retries:
            print(f"\n[DLQ] Action exceeded {self.max_retries} retries")
            print(f"  Strategy: {action.get('strategy')}")
            print(f"  Neighborhood: {action.get('neighborhood')}")
            self.send_to_dlq(action)
            return

        # Exponential backoff
        if retry_count > 0:
            backoff_sec = self.backoff_base ** retry_count
            print(f"\n[Retry {retry_count}/{self.max_retries}] "
                  f"Waiting {backoff_sec}s before retry...")
            time.sleep(backoff_sec)

        # Execute action
        success = self.execute_action(action)

        if success:
            print(f"  ✅ Action succeeded")
            self.stats['succeeded'] += 1
        else:
            # Retry
            print(f"  ❌ Action failed, scheduling retry...")
            action['retry_count'] = retry_count + 1
            action['last_attempt'] = datetime.utcnow().isoformat()

            self.producer.send('iec-action-replay', action)
            self.stats['retried'] += 1

        # Log stats periodically
        if self.stats['processed'] % 10 == 0:
            self.print_stats()

    def execute_action(self, action: dict) -> bool:
        """Execute action via FastAPI ML service.

        Args:
            action: Action dict with strategy and parameters

        Returns:
            True if successful, False otherwise
        """
        strategy = action.get('iec_strategy') or action.get('strategy')

        if not strategy:
            print(f"  ⚠ No strategy in action")
            return False

        # If strategy is do_nothing, acknowledge and skip (already handled)
        if strategy == 'do_nothing' or strategy == 'none':
            print(f"  ✅ Strategy '{strategy}' — already handled, skipping")
            return True

        # Construct endpoint URL
        endpoint = f"{self.ml_service_url}/api/strategy/{strategy}"

        # Prepare request payload
        payload = {
            'strategy': strategy,
            'meta_metrics': action.get('meta_metrics', {}),
            'neighborhood': action.get('neighborhood', 'unknown')
        }

        try:
            # Execute via FastAPI
            response = requests.post(
                endpoint,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"  Strategy: {strategy} -> {result.get('status')}")
                return True
            else:
                print(f"  HTTP {response.status_code}: {response.text[:100]}")
                return False

        except requests.exceptions.Timeout:
            print(f"  ⏱ Timeout")
            return False

        except requests.exceptions.ConnectionError:
            print(f"  🔌 Connection error (ML service down?)")
            return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def send_to_dlq(self, action: dict):
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

        print(f"  ☠️ Sent to DLQ: {action.get('strategy')}")

    def print_stats(self):
        """Print worker statistics."""
        print(f"\n{'='*60}")
        print(f"Action Replay Worker Stats")
        print(f"{'='*60}")
        print(f"Processed: {self.stats['processed']}")
        print(f"Succeeded: {self.stats['succeeded']}")
        print(f"Retried: {self.stats['retried']}")
        print(f"DLQ: {self.stats['dlq']}")
        print(f"{'='*60}\n")


def main():
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
