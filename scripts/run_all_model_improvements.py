"""
Master orchestration script to execute Phase 3, Phase 4, Phase 5 & 6 sequentially.
Runs each pipeline cleanly with proper memory cleanup between stages.
"""

import subprocess
import sys
import time

def run_stage(name, script_path):
    print(f"\n========================================================")
    print(f"STARTING STAGE: {name}")
    print(f"Script: {script_path}")
    print(f"========================================================")
    t0 = time.time()
    res = subprocess.run([sys.executable, "-u", script_path], capture_output=False)
    dur = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED: {name} exited with code {res.returncode} after {dur:.1f}s")
        sys.exit(res.returncode)
    else:
        print(f"COMPLETED: {name} in {dur:.1f}s")

if __name__ == "__main__":
    run_stage("Phase 3: EATD Text Improvement", "scripts/train_eatd_text_improvement.py")
    run_stage("Phase 4: EATD Voice Improvement", "scripts/train_eatd_voice_improvement.py")
    run_stage("Phases 5 & 6: AVDD Visual & Multimodal Fusion", "scripts/train_avdd_visual_and_multimodal.py")
    print("\nALL IMPROVEMENT PIPELINES COMPLETED SUCCESSFULLY!")
