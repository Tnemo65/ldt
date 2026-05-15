#!/usr/bin/env python3
"""
MemStream Redis Loss Chaos Test (Phase 5B).

This test verifies that MemStream handles Redis loss gracefully:
1. Simulates Redis connection loss
2. Verifies IEC degrades silently (no hard errors)
3. Monitors beta staleness increases
4. Verifies scoring continues with degraded quality

Usage:
    python test_redis_loss.py
    python test_redis_loss.py --redis-host redis --redis-port 6379
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaos import MemStreamChaosEngine, FailureScenario


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MemStream Redis Loss Test')
    parser.add_argument(
        '--redis-host',
        default=os.getenv('REDIS_HOST', 'redis'),
        help='Redis host to disconnect'
    )
    parser.add_argument(
        '--redis-port',
        type=int,
        default=int(os.getenv('REDIS_PORT', '6379')),
        help='Redis port'
    )
    parser.add_argument(
        '--kill-duration',
        type=int,
        default=60,
        help='Duration to keep Redis down (seconds)'
    )
    parser.add_argument(
        '--verify-duration',
        type=int,
        default=60,
        help='Duration to verify degradation (seconds)'
    )
    parser.add_argument(
        '--output',
        help='Output file for test report'
    )
    args = parser.parse_args()

    engine = MemStreamChaosEngine()

    print("=" * 70)
    print("MemStream Redis Loss Chaos Test")
    print("=" * 70)
    print(f"Redis Host: {args.redis_host}:{args.redis_port}")
    print(f"Kill Duration: {args.kill_duration}s")
    print(f"Verify Duration: {args.verify_duration}s")
    print()

    result = engine.run_redis_loss_test(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        kill_duration_seconds=args.kill_duration,
        verify_duration_seconds=args.verify_duration
    )

    print()
    print("Test Result:")
    print(f"  Injection Success: {result.injection_success}")
    print(f"  Expected Behavior Verified: {result.expected_behavior_verified}")
    print(f"  Actual Behavior: {result.actual_behavior}")
    print(f"  Alerts Triggered: {', '.join(result.alerts_triggered) or 'None'}")
    print(f"  Duration: {result.duration_seconds:.1f}s" if result.duration_seconds else "  Duration: N/A")

    if result.error_message:
        print(f"  Error: {result.error_message}")

    # For Redis loss, we expect silent degradation (no critical alerts)
    if result.expected_behavior_verified:
        print()
        print("SUCCESS: System degraded silently as expected.")
        print("  - Beta staleness increased")
        print("  - No critical errors or HMAC failures")
        print("  - Scoring continued with degraded IEC quality")

    report = engine.generate_report()
    print()
    print(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")

    return 0 if result.expected_behavior_verified else 1


if __name__ == '__main__':
    exit(main())
