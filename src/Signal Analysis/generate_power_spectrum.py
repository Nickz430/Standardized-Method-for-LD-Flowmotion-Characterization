"""Generate power spectrum comparison figure (Figure4.tif).

This script computes the average power spectrum for both healthy and septic
signals and overlays them on the same plot with frequency boundaries marked.
"""
from pathlib import Path
from typing import Sequence
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from signal_processing_functions import cwt_spectrogram


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found in dataframe: {candidates}")


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_file = repo_root / 'data' / 'filtered' / 'signals_filtered.xlsx'

    if not data_file.exists():
        raise FileNotFoundError(f'Expected data file at {data_file}')

    df = pd.read_excel(data_file)
    data_reperfusion = df[df['Time'] > 213.7].reset_index(drop=True)

    # map candidate column names
    healthy_candidates = ['healthy_subject']
    sepsis_candidates = ['septic_patient']

    healthy_col = _pick_column(data_reperfusion, healthy_candidates)
    sepsis_col = _pick_column(data_reperfusion, sepsis_candidates)

    # Extract signals
    signal_healthy = data_reperfusion[healthy_col].dropna().values
    signal_septic = data_reperfusion[sepsis_col].dropna().values

    fs = 10.0
    frequency_intervals = np.logspace(np.log10(0.001), np.log10(1.5), 200)

    # Compute CWT for healthy
    print("Computing CWT for healthy subject...")
    t0 = time.time()
    power_healthy, times_h, frequencies_h, coif_h = cwt_spectrogram(
        signal_healthy, fs, frequency_intervals, signal_detrend=True, normalize=True
    )
    print(f"  Done in {time.time() - t0:.2f}s")

    # Compute CWT for septic
    print("Computing CWT for septic shock patient...")
    t0 = time.time()
    power_septic, times_s, frequencies_s, coif_s = cwt_spectrogram(
        signal_septic, fs, frequency_intervals, signal_detrend=True, normalize=True
    )
    print(f"  Done in {time.time() - t0:.2f}s")

    # Extract average power in frequency bands
    freq_filter_h = (frequencies_h >= 0.005) & (frequencies_h <= 1.5)
    freq_filter_s = (frequencies_s >= 0.005) & (frequencies_s <= 1.5)

    avg_power_h = np.mean(power_healthy[freq_filter_h, :], axis=1)
    avg_power_s = np.mean(power_septic[freq_filter_s, :], axis=1)

    freqs_h_sel = frequencies_h[freq_filter_h]
    freqs_s_sel = frequencies_s[freq_filter_s]

    # Frequency bounds and labels
    fBounds = [1.6, 0.4, 0.15, 0.06, 0.02, 0.0095]
    fNames = ["Cardiac", "Respiratory", "Myogenic", "Neurogenic", "Endothelial"]

    # Create figure with single axis
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Compute max power for scaling
    max_power = max(np.max(avg_power_h), np.max(avg_power_s))
    min_power_h = np.min(avg_power_h[avg_power_h > 0]) if np.any(avg_power_h > 0) else 0
    min_power_s = np.min(avg_power_s[avg_power_s > 0]) if np.any(avg_power_s > 0) else 0

    # Plot both signals on the same axis with shaded areas
    ax.plot(freqs_h_sel, avg_power_h, color='#1f77b4', linewidth=2.5, label='Healthy Subject')
    ax.fill_betweenx(avg_power_h, 0, freqs_h_sel, color='#1f77b4', alpha=0.15)
    
    ax.plot(freqs_s_sel, avg_power_s, color='darkred', linewidth=2.5, label='Septic Shock Patient')
    ax.fill_betweenx(avg_power_s, 0, freqs_s_sel, color='darkred', alpha=0.15)

    # Add frequency band labels at appropriate heights
    y_label_pos = max_power * 0.08
    for i in range(len(fBounds) - 1):
        f_left = fBounds[i + 1]
        ax.text(f_left * 1.1, y_label_pos, fNames[i], ha='left', va='bottom',
                fontsize=12, color='darkred', fontweight='bold', rotation=0)

    # Add vertical lines for frequency bounds
    ax.vlines(fBounds, ymin=0, ymax=max_power * 1.2,
              color='red', linestyle=':', linewidth=2)

    ax.set_ylabel('Average Power', fontsize=12)
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_xscale('log')
    ax.set_ylim(0, max_power * 1.1)
    ax.set_xlim(0.005, 1.5)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_title('Power Spectrum Comparison: Healthy Subject vs Septic Shock Patient', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Determine output directory
    output_dir = Path(os.environ.get('FLOWMOTION_OUTPUT_DIR', script_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outpath = output_dir / 'Figure4.tif'
    plt.savefig(outpath, dpi=150, format='tiff')
    plt.close()
    print(f'Wrote {outpath}')


if __name__ == '__main__':
    main()
