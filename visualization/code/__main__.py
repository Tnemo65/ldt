"""
Visualization Runner
====================
Run all 8 figure scripts individually or as a batch.

Usage:
    python __main__.py           # run all figures
    python __main__.py 1 2 3    # run specific figures by number
    python __main__.py list      # list available figures
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

FIGURES = {
    '1': ('fig1_monthly_record_volume.py',    '1_Monthly_Record_Volume.png'),
    '2': ('fig2_violation_breakdown.py',       '2_Violation_Breakdown.png'),
    '3': ('fig3_violation_rate_by_vendor.py',  '3_Violation_Rate_by_VendorID.png'),
    '5': ('fig5_distance_vs_fare.py',          '5_Distance_vs_Fare.png'),
    '6': ('fig6_distance_vs_duration.py',      '6_Distance_vs_Duration.png'),
    'F3': ('figF3_normal_vs_special.py',       'F3_normal_vs_special.png'),
    'T5': ('figT5_violations_by_month.py',     'T5_violations_by_month.png'),
    'T7': ('figT7_normal_vs_special.py',       'T7_normal_vs_special_copy.png'),
}


def list_figures():
    print("Available figures:")
    for key, (_, name) in sorted(FIGURES.items()):
        print(f"  {key:>3}: {name}")
    print(f"\nTotal: {len(FIGURES)} figures")


def run_figure(key):
    script, _ = FIGURES[key]
    path = SCRIPT_DIR / script
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('=' * 60)
    result = subprocess.run([sys.executable, str(path)],
                           capture_output=False)
    return result.returncode == 0


def run_all():
    success, failed = [], []
    for key in FIGURES:
        ok = run_figure(key)
        (success if ok else failed).append(key)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('=' * 60)
    print(f"  Succeeded: {len(success)}/{len(FIGURES)}")
    print(f"  Failed:    {len(failed)}/{len(FIGURES)}")
    if failed:
        print(f"  Failed keys: {failed}")


def main():
    if len(sys.argv) == 1 or sys.argv[1] == 'all':
        run_all()
    elif sys.argv[1] == 'list':
        list_figures()
    else:
        for arg in sys.argv[1:]:
            if arg in FIGURES:
                run_figure(arg)
            else:
                print(f"Unknown figure key: {arg}")
                list_figures()
                sys.exit(1)


if __name__ == '__main__':
    main()
