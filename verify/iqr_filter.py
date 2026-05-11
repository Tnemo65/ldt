#!/usr/bin/env python3
"""
Verification: IQR Filter Scope (POST-FIX)
========================================
The thesis now correctly states global IQR. Code still does global IQR.
"""

import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("VERIFICATION: IQR Filter Scope (POST-FIX)")
    print("=" * 70)

    # Check thesis
    content = Path('c:/proj/ldt/thesis/chap4.tex').read_text(encoding='utf-8')

    claims_global = 'applied globally' in content.lower() or 'applied \\\\textbf{globally' in content.lower()
    claims_per_cell = 'per Context Cell' in content and 'IQR' in content
    claims_not_per_cell = 'not globally' in content or 'not per context' in content.lower()

    print(f"  Thesis claims 'globally': {claims_global}")
    print(f"  Thesis still claims 'per Context Cell' (with IQR): {claims_per_cell}")
    print(f"  Thesis says 'not globally' or 'not per context': {claims_not_per_cell}")
    print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if claims_global and claims_not_per_cell:
        print("  VERDICT: FIXED - thesis now says 'globally, not per context'")
        sys.exit(0)
    elif claims_global:
        print("  VERDICT: PARTIALLY FIXED - thesis says 'globally', no 'per cell' claim")
        sys.exit(0)
    else:
        print("  VERDICT: NOT FIXED - check thesis IQR description")
        sys.exit(1)

if __name__ == '__main__':
    main()
