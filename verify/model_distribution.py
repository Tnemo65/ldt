#!/usr/bin/env python3
"""
Verification: Model Distribution Architecture (POST-FIX)
====================================================
The protocol box now correctly describes the MinIO -> load_model_to_broadcast.py -> Kafka flow.
"""

import sys
from pathlib import Path

def check_protocol_box(path: Path) -> dict:
    content = path.read_text(encoding='utf-8', errors='replace')
    lines = content.splitlines()

    in_section = False
    steps = []
    for line in lines:
        if 'Zero-Pipeline-Interruption' in line:
            in_section = True
        if in_section:
            if line.strip().startswith('\\item') or (line.strip().startswith('\\begin{enumerate}')):
                steps.append(line.strip()[:120])
            if 'Atomic model swap' in line or '\\end{enumerate}' in line:
                in_section = False

    return {'protocol_steps': steps}

def main():
    print("=" * 70)
    print("VERIFICATION: Model Distribution (POST-FIX)")
    print("=" * 70)

    result = check_protocol_box(Path('c:/proj/ldt/thesis/chap4.tex'))
    print("Protocol steps:")
    for s in result['protocol_steps']:
        print(f"  {s}")
    print()

    # Check step 3 mentions MinIO/bridge
    step3_ok = any('MinIO' in s or 'minio' in s or 'load_model' in s for s in result['protocol_steps'])

    # Check no step says "MLflow publishes model bytes directly"
    step4_redundant = False
    content = Path('c:/proj/ldt/thesis/chap4.tex').read_text(encoding='utf-8')
    if 'MLflow publishes model bytes to' in content and 'Zero-Pipeline' in content:
        # Check if it's in the protocol box
        idx = content.find('Zero-Pipeline-Interruption')
        end = content.find('Atomic model swap', idx)
        box_content = content[idx:end] if idx >= 0 and end >= 0 else ""
        if 'MLflow publishes model bytes' in box_content:
            step4_redundant = True

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"  Step mentions MinIO/bridge:  {step3_ok}")
    print(f"  Redundant 'MLflow publishes bytes' step: {step4_redundant}")

    if step3_ok and not step4_redundant:
        print("  VERDICT: FIXED - protocol box correctly describes the flow")
        sys.exit(0)
    else:
        print("  VERDICT: Still has issues")
        sys.exit(1)

if __name__ == '__main__':
    main()
