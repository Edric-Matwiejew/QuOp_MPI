#!/usr/bin/env python3
"""
Integration test runner for QuOp_MPI examples.

Runs the actual example scripts and validates that the optimization 
results fall within expected bounds.

Usage:
    python run_integration_tests.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
EXPECTED_RESULTS_FILE = SCRIPT_DIR / "expected_results.json"


def load_expected_results():
    with open(EXPECTED_RESULTS_FILE, "r") as f:
        return json.load(f)


def run_example(script_path, cwd, timeout=300):
    """Run an example script and return parsed results."""
    result = subprocess.run(
        ["mpiexec", "-n", "1", "python", script_path],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout
    )
    output = result.stdout + result.stderr
    
    # Parse all fun values from output (benchmark produces multiple)
    fun_matches = re.findall(r'^fun:\s+([-\d.e+]+)', output, re.MULTILINE)
    norm_matches = re.findall(r'final state norm:\s+([\d.e+-]+)', output, re.MULTILINE)
    
    funs = [float(f) for f in fun_matches] if fun_matches else []
    norms = [float(n) for n in norm_matches] if norm_matches else []
    
    return {
        "success": result.returncode == 0,
        "funs": funs,
        "norms": norms,
        "output": output,
        "returncode": result.returncode
    }


def validate_result(result, config):
    """Validate test result against expected bounds."""
    errors = []
    
    if not result["funs"]:
        errors.append("No 'fun' values found in output")
    else:
        # Check the last fun value (final result)
        fun = result["funs"][-1]
        if fun < config["fun_min"]:
            errors.append(f"fun={fun:.6f} < fun_min={config['fun_min']:.6f}")
        elif fun > config["fun_max"]:
            errors.append(f"fun={fun:.6f} > fun_max={config['fun_max']:.6f}")
    
    if not result["norms"]:
        errors.append("No 'norm' values found in output")
    else:
        norm = result["norms"][-1]
        if norm < config["norm_min"]:
            errors.append(f"norm={norm:.6f} < norm_min={config['norm_min']:.6f}")
    
    return errors


def main():
    expected = load_expected_results()
    
    print(f"Running {len(expected)} integration tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, config in expected.items():
        example_dir = EXAMPLES_DIR / config["example_dir"]
        script = config["script"]
        
        print(f"\n[{test_name}] {config['description']}...")
        
        try:
            result = run_example(script, example_dir)
            errors = validate_result(result, config)
            
            if result["success"] and not errors:
                fun = result["funs"][-1] if result["funs"] else None
                norm = result["norms"][-1] if result["norms"] else None
                print(f"  PASSED (fun={fun:.6f}, norm={norm:.6f})")
                passed += 1
            else:
                print("  FAILED")
                if not result["success"]:
                    print(f"    Execution failed (returncode={result['returncode']})")
                for err in errors:
                    print(f"    - {err}")
                failed += 1
        except subprocess.TimeoutExpired:
            print("  FAILED (timeout)")
            failed += 1
        except Exception as e:
            print(f"  FAILED ({e})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(expected)} tests")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
