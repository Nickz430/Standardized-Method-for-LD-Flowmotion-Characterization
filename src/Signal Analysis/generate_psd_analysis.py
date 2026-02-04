"""Generate PSD analysis figure with power spectrum and PSD values.

This script computes and plots the average power spectrum with PSD values
displayed on the frequency bands for both healthy and septic subjects.
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

    # PSD values for each frequency band
    psd_healthy = [2110.5, 4444.3, 2882.4, 954.6, 162.8]
    psd_septic = [232.5, 73.5, 17.4, 7.4, 1.2]

    # Create figure with two separate subplots
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [4, 4]})

    # ===== Plot 1: Healthy Subject =====
    ax[0].plot(freqs_h_sel, avg_power_h, color='black', linewidth=2)
    ax[0].fill_betweenx(avg_power_h, 0, freqs_h_sel, color='gray', alpha=0.5)

    # Add frequency band labels with PSD values
    for i in range(len(fBounds) - 1):
        f_left = fBounds[i + 1]
        label = f"{fNames[i]}\nPSD = {psd_healthy[i]:.1f}"
        ax[0].text(f_left * 1.05,
                   np.max(avg_power_h),
                   label, ha='left', va='bottom', fontsize=11, color='darkred')

    # Add vertical lines for frequency bounds
    ax[0].vlines(fBounds, ymin=0, ymax=np.max(avg_power_h) * 1.2,
                 color='red', linestyle=':', linewidth=2)

    ax[0].set_ylabel('Average Power', fontsize=12)
    ax[0].set_xlabel('Frequency [Hz]', fontsize=12)
    ax[0].set_xscale('log')
    ax[0].set_ylim(0, np.max(avg_power_h) * 1.2)
    ax[0].set_xlim(0.005, 1.5)
    ax[0].set_xticks([0.01, 0.1, 1.0])
    ax[0].set_title('Healthy Subject', fontsize=14, fontweight='bold')
    ax[0].grid(True, alpha=0.3)

    # ===== Plot 2: Septic Shock Patient =====
    ax[1].plot(freqs_s_sel, avg_power_s, color='black', linewidth=2)
    ax[1].fill_betweenx(avg_power_s, 0, freqs_s_sel, color='gray', alpha=0.5)

    # Add frequency band labels with PSD values
    for i in range(len(fBounds) - 1):
        f_left = fBounds[i + 1]
        label = f"{fNames[i]}\nPSD = {psd_septic[i]:.1f}"
        ax[1].text(f_left * 1.05,
                   np.max(avg_power_s),
                   label, ha='left', va='bottom', fontsize=11, color='darkred')

    # Add vertical lines for frequency bounds
    ax[1].vlines(fBounds, ymin=0, ymax=np.max(avg_power_s) * 1.2,
                 color='red', linestyle=':', linewidth=2)

    ax[1].set_ylabel('Average Power', fontsize=12)
    ax[1].set_xlabel('Frequency [Hz]', fontsize=12)
    ax[1].set_xscale('log')
    ax[1].set_ylim(0, np.max(avg_power_s) * 1.2)
    ax[1].set_xlim(0.005, 1.5)
    ax[1].set_xticks([0.01, 0.1, 1.0])
    ax[1].set_title('Septic Shock Patient', fontsize=14, fontweight='bold')
    ax[1].grid(True, alpha=0.3)

    # Final adjustments
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    
    # Determine output directory
    output_dir = Path(os.environ.get('FLOWMOTION_OUTPUT_DIR', script_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outpath = output_dir / 'Figure5_PSD.tif'
    plt.savefig(outpath, dpi=300, format='tiff')
    plt.close()
    print(f'Wrote {outpath}')


if __name__ == '__main__':
    main()
