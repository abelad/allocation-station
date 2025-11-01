#!/usr/bin/env python
"""Test all example scripts to ensure they run without errors."""

import os
import subprocess
import sys
from pathlib import Path

def test_all_examples():
    """Run all example scripts and report results."""
    examples_dir = Path("examples")
    scripts = sorted([f for f in examples_dir.glob("*.py")])

    results = []
    print("=" * 80)
    print("TESTING ALL EXAMPLE SCRIPTS")
    print("=" * 80)

    for script in scripts:
        print(f"\nTesting {script.name}...")
        print("-" * 40)

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"[OK] {script.name} completed successfully")
                results.append((script.name, True))
            else:
                print(f"[FAILED] {script.name} failed with error:")
                print(result.stderr[:500] if result.stderr else result.stdout[:500])
                results.append((script.name, False))

        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {script.name} timed out after 30 seconds")
            results.append((script.name, False))
        except Exception as e:
            print(f"[ERROR] {script.name} failed with exception: {e}")
            results.append((script.name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for script_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {script_name:40s} {status}")

    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} scripts passed")
    print("=" * 80)

    return passed == total

if __name__ == "__main__":
    success = test_all_examples()
    sys.exit(0 if success else 1)