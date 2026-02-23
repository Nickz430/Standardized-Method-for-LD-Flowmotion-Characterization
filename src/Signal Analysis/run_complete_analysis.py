"""
Master Script - Complete Laser Doppler Flowmotion Analysis Pipeline

This script orchestrates the complete signal processing and analysis workflow in sequence:
    1. Hampel Filter: Applies robust filtering to remove outliers/artifacts
    2. Signal Vasomotion Area: Computes envelope analysis and vasomotion metrics
    3. Spectrograms: Generates CWT-based time-frequency representations
    4. PSD Analysis: Extracts Power Spectral Density across frequency bands
    5. Power Spectrum: Comparison plots of healthy vs septic power spectra
    6. CWT Characteristics: Comprehensive time and spectral domain feature extraction

All steps are logged with timing information and status indicators.

Author: Signal Processing Pipeline
Date: 2026
"""

import subprocess
import time
import os
from pathlib import Path
from datetime import datetime


def print_header(title: str, level: int = 1) -> None:
    """
    Print formatted section headers.
    
    Parameters
    ----------
    title : str
        Title text to display
    level : int
        Header level (1=main, 2=section)
    """
    width = 100 if level == 1 else 80
    char = "=" if level == 1 else "-"
    print("\n" + char * width)
    print(f"  {title.upper()}")
    print(char * width + "\n")


def print_step(step_num: int, step_name: str) -> None:
    """
    Print step information with counter.
    
    Parameters
    ----------
    step_num : int
        Step number in sequence
    step_name : str
        Descriptive name of the step
    """
    print(f"\n[{step_num}/6] {step_name}")
    print("-" * 60)


def run_script(script_path: Path, script_name: str, step_num: int, repo_root: Path) -> bool:
    """
    Execute a Python script and monitor execution.
    
    Parameters
    ----------
    script_path : Path
        Full path to the script to execute
    script_name : str
        Display name of the script
    step_num : int
        Step number in pipeline
    
    Returns
    -------
    bool
        True if execution successful, False otherwise
    """
    print_step(step_num, script_name)
    
    if not script_path.exists():
        print(f"✗ ERROR: Script not found at {script_path}")
        return False
    
    print(f"Starting: {script_path.name}")
    start_time = time.time()
    
    try:
        # Set output directory environment variable
        env = os.environ.copy()
        env['FLOWMOTION_OUTPUT_DIR'] = str(repo_root / 'plt')
        
        # Execute script
        result = subprocess.run(
            [r"python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout per script
            env=env
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {script_name} completed in {elapsed:.2f}s")
            if result.stdout:
                # Print last few lines of output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
            return True
        else:
            print(f"✗ FAILED: {script_name} (Exit code: {result.returncode})")
            print(f"Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {script_name} exceeded 10 minutes")
        return False
    except Exception as e:
        print(f"✗ ERROR: {script_name} - {str(e)}")
        return False


def main():
    """
    Execute the complete analysis pipeline in sequence.
    
    Workflow:
        1. Data Filtering: Hampel filtering for outlier removal
        2. Feature Analysis: Signal vasomotion and envelope metrics
        3. Time-Frequency: Spectrogram generation using CWT
        4. Spectral Metrics: PSD analysis across frequency bands
        5. Spectral Comparison: Power spectrum overlay plots
    """
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    print_header("LASER DOPPLER FLOWMOTION - COMPLETE ANALYSIS PIPELINE")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {repo_root}")
    
    # Helper to locate scripts in the repo (allows flexible layout)
    def find_script(script_name: str) -> Path:
        candidate = script_dir / script_name
        if candidate.exists():
            return candidate

        matches = list(repo_root.rglob(script_name))
        if matches:
            return matches[0]

        candidate2 = repo_root / 'src' / script_name
        if candidate2.exists():
            return candidate2

        return candidate

    # Define pipeline steps using script file names; paths will be resolved dynamically
    pipeline_steps = [
        ("hampel_filter.py", "Step 1: Hampel Outlier Filtering", 1),
        ("signal_VA_analysis.py", "Step 2: Signal Vasomotion Area Analysis", 2),
        ("generate_spectrograms.py", "Step 3: Spectrogram Generation (CWT)", 3),
        ("generate_psd_analysis.py", "Step 4: PSD Analysis", 4),
        ("generate_power_spectrum.py", "Step 5: Power Spectrum Comparison", 5),
    ]
    
    # Execute pipeline
    print_header("EXECUTION PIPELINE", level=2)
    
    results = {}
    total_start = time.time()
    
    for script_name, display_name, step_num in pipeline_steps:
        script_path = find_script(script_name)
        if script_path.exists():
            resolved_note = f"Resolved '{script_name}' -> {script_path}"
            print(resolved_note)
        else:
            print(f"Looking for '{script_name}' at {script_path} (not found)")

        success = run_script(script_path, display_name, step_num, repo_root)
        results[display_name] = success
        
        if not success:
            print(f"\n⚠ Pipeline halted at step {step_num}")
            print("Review the error above and rerun after fixing the issue.")
            break
    
    # Summary report
    print_header("EXECUTION SUMMARY", level=2)
    total_time = time.time() - total_start
    
    completed = sum(1 for v in results.values() if v)
    total_steps = len(results)
    
    print("Pipeline Status:")
    print("-" * 60)
    for step_name, success in results.items():
        status = "✓ COMPLETE" if success else "✗ FAILED"
        print(f"  {status:15} {step_name}")
    
    print("\n" + "=" * 100)
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Steps Completed: {completed}/{total_steps}")
    
    if completed == total_steps:
        print("\n✓ ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated Outputs:")
        print("\n  Figures (saved to plt/ folder):")
        print("    - Figure_1.tif: Full signal comparison (Baseline, Ischemic, Reperfusion)")
        print("    - Figure2.tif: Signal envelopes and vasomotion area")
        print("    - Figure3A.tif: Healthy subject spectrogram")
        print("    - Figure3B.tif: Septic patient spectrogram")
        print("    - Figure4.tif: Power spectrum comparison")
        print("    - Figure5_PSD.tif: PSD values on power spectra")

    else:
        print("\n✗ PIPELINE INCOMPLETE - CHECK ERRORS ABOVE")
    
    print("\n" + "=" * 100)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return completed == total_steps


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
