#!/usr/bin/env python3
"""
Verification: Algorithm Naming (POST-FIX)
=====================================
Checks if sklearn IsolationForest is correctly named throughout.
"""

import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("VERIFICATION: Algorithm Naming (POST-FIX)")
    print("=" * 70)

    # Check code: if_scoring_operator.py uses pickle.loads (no direct import)
    # The sklearn IsolationForest is trained in train_iforest.py / train_model.py
    op_content = Path('c:/proj/ldt/src/operators/if_scoring_operator.py').read_text(encoding='utf-8')
    train_content = Path('c:/proj/ldt/src/ml/train_iforest.py').read_text(encoding='utf-8')
    deploy_content = Path('c:/proj/ldt/deployment/scripts/train_model.py').read_text(encoding='utf-8')

    code_ok = (
        'iForestASD' not in op_content
        and 'sklearn IsolationForest' in op_content
        and 'from sklearn.ensemble import IsolationForest' in train_content
        and 'from sklearn.ensemble import IsolationForest' in deploy_content
    )
    print(f"  if_scoring_operator.py has 'sklearn IsolationForest': {'sklearn IsolationForest' in op_content}")
    print(f"  if_scoring_operator.py has NO 'iForestASD': {'iForestASD' not in op_content}")
    print(f"  train_iforest.py uses sklearn: {'IsolationForest' in train_content}")
    print(f"  train_model.py uses sklearn: {'IsolationForest' in deploy_content}")
    print()

    # Check thesis
    import glob
    matches = glob.glob('c:/proj/ldt/thesis/**/*.tex', recursive=True)
    has_sHST = False
    for path in matches:
        content = Path(path).read_text(encoding='utf-8', errors='replace')
        if 'sHST' in content:
            has_sHST = True
            print(f"  sHST found in: {path}")

    print(f"  Thesis has 'sHST': {has_sHST}")
    print()

    print("=" * 70)
    if code_ok and not has_sHST:
        print("VERDICT: PASS - sklearn IsolationForest used consistently")
        sys.exit(0)
    else:
        print("VERDICT: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    main()
