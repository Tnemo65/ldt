#!/usr/bin/env python3
"""
Verification: Lock Cooldown + Action Replay Worker (POST-FIX)
=========================================================
Checks if the thesis has sufficient detail on lock/cooldown mechanism.
"""

import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("VERIFICATION: Lock Cooldown (POST-FIX)")
    print("=" * 70)

    content = Path('c:/proj/ldt/thesis/chap4.tex').read_text(encoding='utf-8')

    checks = {
        'exponential backoff formula': '$2^n' in content or '2^n' in content,
        'max 10 retries': '10' in content and 'retry' in content.lower(),
        'DLQ topic': 'iec-action-dlq' in content,
        'Action Replay Worker': 'Action Replay Worker' in content,
        'lock release': 'lock release' in content.lower(),
        'Strategy fallback': 'Strategy' in content and 'fallback' in content.lower(),
    }

    for check, result in checks.items():
        print(f"  {check}: {result}")

    print()
    print("=" * 70)
    if all(checks.values()):
        print("VERDICT: PASS - thesis has all lock/cooldown details")
        sys.exit(0)
    else:
        missing = [k for k, v in checks.items() if not v]
        print(f"VERDICT: FAIL - missing: {missing}")
        sys.exit(1)

if __name__ == '__main__':
    main()
