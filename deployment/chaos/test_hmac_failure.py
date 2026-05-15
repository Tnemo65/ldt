#!/usr/bin/env python3
"""
MemStream Model HMAC Failure Chaos Test (Phase 5B).

This test verifies that MemStream handles model HMAC verification failures:
1. Injects wrong HMAC signature into checkpoint
2. Verifies MemStream_ModelHMACFailure alert fires
3. Verifies system fails hard (no silent degradation)
4. Ensures security incident is properly flagged

Usage:
    python test_hmac_failure.py
    python test_hmac_failure.py --signing-key my-secret-key
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaos import MemStreamChaosEngine, FailureScenario


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MemStream Model HMAC Failure Test')
    parser.add_argument(
        '--signing-key',
        default=os.getenv('MEMSTREAM_MODEL_SIGNING_KEY'),
        help='Model signing key (used for verification, not for injection)'
    )
    parser.add_argument(
        '--verify-duration',
        type=int,
        default=30,
        help='Duration to verify hard failure (seconds)'
    )
    parser.add_argument(
        '--output',
        help='Output file for test report'
    )
    args = parser.parse_args()

    engine = MemStreamChaosEngine()

    print("=" * 70)
    print("MemStream Model HMAC Failure Chaos Test")
    print("=" * 70)
    print(f"Verify Duration: {args.verify_duration}s")
    print()

    result = engine.run_hmac_failure_test(
        signing_key=args.signing_key,
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

    # For HMAC failure, we expect hard failure with security alert
    if result.expected_behavior_verified:
        print()
        print("SUCCESS: System failed hard with security alert as expected.")
        print("  - Critical security alert fired")
        print("  - HMAC verification blocked loading")
        print("  - System in degraded/failsafe mode")

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
