"""Generate spectrogram figures (Figure3A.tif and Figure3B.tif).

This script uses `cwt_spectrogram` and `spectrogram_plot` from
`signal_processing_functions.py` to compute and plot time-frequency
representations for two example signals in the provided dataset.
"""
from pathlib import Path
from typing import Sequence
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from signal_processing_functions import cwt_spectrogram, spectrogram_plot


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found in dataframe: {candidates}")


def plot_spectrogram_for_column(df: pd.DataFrame, col_name: str, outpath: Path, title: str,
                               norm: LogNorm, is_healthy: bool) -> None:
    """Plot spectrogram using a shared normalization based on healthy subject's power.
    
    Parameters
    - df: dataframe with signals
    - col_name: column name to extract
    - outpath: where to save the figure
    - title: plot title
    - norm: shared LogNorm object computed from healthy subject's power
    - is_healthy: True if this is the healthy subject (used for color)
    """
    sig = df[col_name].dropna().values
    fs = 10.0
    n_samples = sig.size
    total_duration = n_samples / fs
    times_vec = np.linspace(0, total_duration, n_samples)

    # frequency grid used for the CWT (Hz)
    frequency_intervals = np.logspace(np.log10(0.001), np.log10(1.5), 200)

    t0 = time.time()
    power, times_cwt, frequencies, coif = cwt_spectrogram(
        sig, fs, frequency_intervals, signal_detrend=True, normalize=True
    )
    print(f"CWT time for {col_name}: {time.time() - t0:.2f}s")

    # select full time range (keeps original behaviour)
    indices = np.where((times_cwt >= times_cwt.min()) & (times_cwt <= times_cwt.max()))[0]
    power_slice = power[:, indices]

    freq_filter = (frequencies >= 0.001) & (frequencies <= 1.5)
    avg_power = np.mean(power_slice[freq_filter, :], axis=1)

    # build axes layout like the original snippet
    fig, ax = plt.subplots(2, 2,
        gridspec_kw={
            'height_ratios': [1, 4],
            'width_ratios': [4, 1]
        },
        figsize=(10, 6)
    )
    plt.subplots_adjust(hspace=0.5)

    color = '#1f77b4' if is_healthy else 'darkred'
    ax[0, 0].plot(sig if sig.size > 0 else [], color=color)
    ax[0, 0].set_title(title)

    # remove empty top-right axis
    fig.delaxes(ax[0, 1])

    fBounds = [1.6, 0.4, 0.15, 0.06, 0.02, 0.0095]
    fNames = ["Cardiac", "Respiratory", "Myogenic", "Neurogenic", "Endothelial"]
    bounds = [fBounds, fNames]

    spectrogram_plot(
        power, times_cwt, frequencies, coif,
        cmap='jet', ax=ax[1, 0], log_y=True, norm=norm, bounds=bounds
    )

    for bound in fBounds:
        ax[1, 0].axhline(bound, color='darkred', linestyle='--', linewidth=1)

    ax[1, 0].set_ylim(0.005, 1.5)

    # right-side average power plot (avg_power vs freq)
    freqs_sel = frequencies[freq_filter]
    ax[1, 1].plot(avg_power[:len(freqs_sel)], freqs_sel, color='black')
    ax[1, 1].fill_betweenx(freqs_sel, 0, avg_power[:len(freqs_sel)], color='grey', alpha=0.5)

    ax[1, 1].set_xlabel('Average Power')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_ylim(0.005, 1.5)
    ax[1, 1].set_yticks([])
    ax[1, 1].set_ylabel('')

    for valley in fBounds:
        ax[1, 1].axhline(valley, color='red', linestyle=':', linewidth=2)

    ax[1, 1].grid(False)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, format='tiff')
    plt.close(fig)
    print(f'Wrote {outpath}')


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_file = repo_root / 'data' / 'filtered' / 'signals_filtered.xlsx'

    if not data_file.exists():
        raise FileNotFoundError(f'Expected data file at {data_file}')

    df = pd.read_excel(data_file)

    # create a short-window dataframe like in your snippet
    data_reperfusion = df[df['Time'] > 213.7].reset_index(drop=True)

    # map candidate column names to what exists in the sheet
    healthy_candidates = ['healthy_subject']
    sepsis_candidates = ['septic_patient']

    healthy_col = _pick_column(data_reperfusion, healthy_candidates)
    sepsis_col = _pick_column(data_reperfusion, sepsis_candidates)

    # === First pass: compute power for healthy subject to get reference normalization ===
    print("Computing reference normalization from healthy subject...")
    sig_healthy = data_reperfusion[healthy_col].dropna().values
    fs = 10.0
    frequency_intervals = np.logspace(np.log10(0.001), np.log10(1.5), 200)

    t0 = time.time()
    power_healthy, _, _, _ = cwt_spectrogram(
        sig_healthy, fs, frequency_intervals, signal_detrend=True, normalize=True
    )
    print(f"  CWT time: {time.time() - t0:.2f}s")

    # Compute normalization based on healthy subject's power
    vmin = np.nanmin(power_healthy)
    vmax = np.nanmax(power_healthy)
    shared_norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.01))
    print(f"  Shared normalization: vmin={vmin:.2e}, vmax={vmax:.2e}")

    # Determine output directory
    output_dir = Path(os.environ.get('FLOWMOTION_OUTPUT_DIR', script_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot healthy spectrogram (Figure3A) with shared norm
    outA = output_dir / 'Figure3A.tif'
    plot_spectrogram_for_column(data_reperfusion, healthy_col, outA, 'Healthy Subject',
                                norm=shared_norm, is_healthy=True)

    # Plot septic spectrogram (Figure3B) with same shared norm
    outB = output_dir / 'Figure3B.tif'
    plot_spectrogram_for_column(data_reperfusion, sepsis_col, outB, 'Septic Shock Patient',
                                norm=shared_norm, is_healthy=False)


if __name__ == '__main__':
    main()
