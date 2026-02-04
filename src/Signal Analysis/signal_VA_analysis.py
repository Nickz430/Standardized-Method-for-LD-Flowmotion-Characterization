"""Generate signals and VA and VAp from the example signals.
"""
from pathlib import Path
from typing import Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import local helpers (this file is expected to live in the same folder)
from signal_processing_functions import find_peaks, find_valleys


def compute_envelopes(sig: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper and lower linear-interpolated envelopes and their difference.

    Uses `find_peaks` and `find_valleys` from `signal_processing_functions.py`.
    Falls back to including the signal endpoints if no extrema are found.
    """
    sig = np.asarray(sig)
    t = np.asarray(t)

    peaks = find_peaks(sig)
    valleys = find_valleys(sig)

    # Ensure there are points for interpolation; include endpoints if necessary
    if peaks.size == 0:
        peaks = np.array([0, sig.size - 1])
    else:
        if peaks[0] != 0:
            peaks = np.concatenate(([0], peaks))
        if peaks[-1] != sig.size - 1:
            peaks = np.concatenate((peaks, [sig.size - 1]))

    if valleys.size == 0:
        valleys = np.array([0, sig.size - 1])
    else:
        if valleys[0] != 0:
            valleys = np.concatenate(([0], valleys))
        if valleys[-1] != sig.size - 1:
            valleys = np.concatenate((valleys, [sig.size - 1]))

    interp_upper = interp1d(t[peaks], sig[peaks], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_lower = interp1d(t[valleys], sig[valleys], kind='linear', bounds_error=False, fill_value='extrapolate')

    upper = interp_upper(t)
    lower = interp_lower(t)
    envelope_diff = upper - lower

    return upper, lower, envelope_diff


def signal_general_plot(df: pd.DataFrame, outpath: Path) -> None:
    """Create Figure_1.tif: full-signal comparison with vertical period markers."""
    plt.figure(figsize=(12, 4))

    # Plot the signals (apply same offset used in your example)
    plt.plot(df['healthy_subject'], label='Healthy Subject', color='#1f77b4', linewidth=2)
    # apply -10 offset to the second series to mimic the original visualization
    plt.plot(df['septic_patient'] - 10, label='Septic Shock Patient', color='darkred', linewidth=2)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Amplitude (BPU)')
    plt.legend(fontsize=12)
    plt.grid(False)

    # Vertical lines marking periods (as in original snippet)
    plt.axvline(x=600, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=2137, color='black', linestyle='--', linewidth=1)

    # Add period labels (positions chosen similarly to your snippet)
    y_max = float(df['healthy_subject'].max())
    plt.text(300, y_max - 50, 'Baseline', ha='center', fontsize=14, fontweight='bold')
    plt.text(1300, y_max - 50, 'Ischemic Period', ha='center', fontsize=14, fontweight='bold')
    plt.text(2700, y_max - 50, 'Reperfusion', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    outpath_parent = outpath.parent
    outpath_parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, format='tiff')
    plt.close()


def va_signal_envelopes_plot(df: pd.DataFrame, outpath: Path) -> None:
    """Create Figure2.tif: envelopes and filled vasomotion area on a short segment.

    This uses data where `Time < 60` (as in the example snippet).
    """
    df_short = df[df['Time'] < 60].reset_index(drop=True)

    signal1 = df_short['healthy_subject'].values
    signal2 = df_short['septic_patient'].values
    t1 = np.arange(signal1.size)
    t2 = np.arange(signal2.size)

    upper1, lower1, diff1 = compute_envelopes(signal1, t1)
    upper2, lower2, diff2 = compute_envelopes(signal2, t2)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Healthy Subject
    axs[0].plot(t1, signal1, label='LD Signal (Healthy Subject)', color='#1f77b4', linewidth=1.5)
    axs[0].plot(t1, upper1, '--', label='Upper Envelope U(t)', color='#1f77b4', linewidth=1)
    axs[0].plot(t1, lower1, '--', label='Lower Envelope L(t)', color='#1f77b4', linewidth=1)
    axs[0].fill_between(t1, lower1, upper1, color='#1f77b4', alpha=0.2, label='Vasomotion Area (VA)')
    axs[0].set_ylabel('Signal Amplitude (BPU)')
    axs[0].set_title('Healthy Subject')
    axs[0].legend()
    axs[0].grid(True)

    # Septic Shock Patient
    axs[1].plot(t2, signal2, label='LD Signal (Septic Shock)', color='darkred', linewidth=1.5)
    axs[1].plot(t2, upper2, '--', label='Upper Envelope U(t)', color='darkred', linewidth=1)
    axs[1].plot(t2, lower2, '--', label='Lower Envelope L(t)', color='darkred', linewidth=1)
    axs[1].fill_between(t2, lower2, upper2, color='darkred', alpha=0.2, label='Vasomotion Area (VA)')
    axs[1].set_xlabel('Time (samples)')
    axs[1].set_ylabel('Signal Amplitude (BPU)')
    axs[1].set_title('Septic Shock Patient')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath_parent = outpath.parent
    outpath_parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, format='tiff')
    plt.close()


def main():
    # Locate data file relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_file = repo_root / 'data' / 'filtered' / 'signals_filtered.xlsx'

    if not data_file.exists():
        raise FileNotFoundError(f'Expected data file at {data_file}')

    df = pd.read_excel(data_file)

    # Determine output directory
    output_dir = Path(os.environ.get('FLOWMOTION_OUTPUT_DIR', script_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1
    figure1_out = output_dir / 'Figure_1.tif'
    signal_general_plot(df, figure1_out)
    print(f'Wrote {figure1_out}')

    # Figure 2
    figure2_out = output_dir / 'Figure2.tif'
    va_signal_envelopes_plot(df, figure2_out)
    print(f'Wrote {figure2_out}')


if __name__ == '__main__':
    main()
