"""
Main Entry Point — Run All Lab Steps
====================================
This script runs the Day 22 lab steps sequentially.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_step(step_file):
    print(f"\n\n{'#'*80}")
    print(f"## RUNNING: {step_file}")
    print(f"{'#'*80}\n")
    
    try:
        # Use sys.executable to ensure we use the same python interpreter
        result = subprocess.run([sys.executable, step_file], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {step_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab Steps")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step (1-4)")
    args = parser.parse_args()

    steps = {
        1: "pseudocode/01_langsmith_rag_pipeline.py",
        2: "pseudocode/02_prompt_hub_ab_routing.py",
        3: "pseudocode/03_ragas_evaluation.py",
        4: "pseudocode/04_guardrails_validator.py"
    }

    if args.step:
        run_step(steps[args.step])
    else:
        print("🚀 Starting full lab execution...")
        for i in range(1, 5):
            success = run_step(steps[i])
            if not success:
                print(f"🛑 Stopping due to error in step {i}")
                break
        else:
            print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()
