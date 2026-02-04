"""Signal processing helper functions used across the project.

This module provides a few focused utilities for time-frequency
analysis (CWT spectrogram), plotting helpers and simple peak/valley
detectors used by downstream scripts.

Functions:
- `cwt_spectrogram` : compute CWT power, times, frequencies and cone-of-influence
- `spectrogram_plot` : simple spectrogram plotting helper
- `find_peaks` / `find_valleys` : small zero-crossing based extrema detectors
- `calculate_areas_trapz` : integrate power over frequency intervals

Notes:
- The small peak/valley detectors are lightweight helpers; for robust
  production use consider `scipy.signal.find_peaks`.
"""

from typing import Sequence, Tuple, List, Optional

import numpy as np
from scipy import ndimage
from scipy.signal import detrend
import pywt
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['figure.figsize'] = (10, 8)


__all__ = [
    'cwt_spectrogram',
    'spectrogram_plot',
    'find_peaks',
    'find_valleys',
    'calculate_areas_trapz',
]


def cwt_spectrogram(x: Sequence[float], fs: float, frequencies: Sequence[float], *,
                    signal_detrend: bool = True, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a continuous wavelet transform (CWT) spectrogram.

    Parameters
    - x: 1-D signal samples.
    - fs: sampling frequency (Hz).
    - frequencies: sequence of center frequencies (Hz) to evaluate.
    - signal_detrend: if True detrend the signal before analysis.
    - normalize: if True scale the signal by 100 (keeps previous behaviour).

    Returns
    - power: 2-D array (scales x times) containing squared magnitude of CWT coef.
    - times: time vector corresponding to samples (seconds).
    - frequencies: frequency vector (Hz) used for rows of `power`.
    - coif: cone-of-influence frequency mask (same length as `times`).

    Notes
    - Uses the complex Morlet (`cmor1.5-1.0`) wavelet. The mapping from
      requested frequencies to scales uses `pywt.central_frequency`.
    """
    x_arr = np.asarray(x)
    N = x_arr.size
    dt = 1.0 / float(fs)
    times = np.arange(N) * dt

    if signal_detrend:
        x_arr = detrend(x_arr)
    if normalize:
        x_arr = x_arr / 100.0

    # compute scales that correspond to requested frequencies
    freqs_req = np.asarray(frequencies)
    if np.any(freqs_req <= 0):
        raise ValueError('frequencies must be positive')

    central = pywt.central_frequency('cmor1.5-1.0')
    scales = central / (freqs_req * dt)

    coef, freqs = pywt.cwt(x_arr, scales, 'cmor1.5-1.0', sampling_period=dt)

    power = np.abs(coef) ** 2
    power = ndimage.gaussian_filter(power, sigma=2)

    # approximate cone of influence for this wavelet (preserves original idea)
    f0 = 2 * np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0 ** 2))
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    coif = 1.0 / coi

    return power, times, freqs, coif


def spectrogram_plot(z: np.ndarray, times: Sequence[float], frequencies: Sequence[float], coif: Sequence[float], *,
                     cmap: Optional[str] = None, norm=None, ax=None, colorbar: bool = True, log_y: bool = False, bounds: Optional[tuple] = None):
    """Plot a spectrogram-like matrix `z` vs `times` and `frequencies`.

    Parameters
    - z: 2-D array (freqs x times) of power or magnitude.
    - times: 1-D time vector (seconds)
    - frequencies: 1-D frequency vector (Hz)
    - coif: cone-of-influence values to overlay
    - cmap: colormap name or None to use default
    - norm: matplotlib Normalize instance (optional)
    - ax: matplotlib Axes to draw on. If None, a new figure/axes created.
    - colorbar: whether to draw a colorbar
    - log_y: set y-axis to log scale
    - bounds: optional tuple (fBounds, fNames) to draw horizontal markers

    Returns: the `ax` used for plotting.
    """
    if cmap is None:
        cmap = get_cmap('Greys')
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    T, F = np.meshgrid(times, frequencies)
    im = ax.pcolormesh(T, F, z, cmap=cmap, norm=norm, shading='auto')
    ax.plot(times, coif, color='k')
    ax.fill_between(times, coif, step="mid", alpha=0.4)

    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4)
        cbar = fig.colorbar(im, cax=cbaxes, orientation='vertical', pad=0.05)
        cbar.set_label('Power [-]')

    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(np.min(frequencies), np.max(frequencies))

    if log_y:
        ax.set_yscale('log')
    if bounds is not None:
        fBounds, fNames = bounds
        for bound, name in zip(fBounds, fNames):
            ax.axhline(bound, color='r', linestyle='--', linewidth=1)
            ax.text(np.min(times) - 0.1, bound * 0.85, name, color='black', va='center')
    return ax


def find_peaks(signal: Sequence[float], min_height: float = 0.0) -> np.ndarray:
    """Return sample indices of local maxima using simple zero-crossing of derivative.

    This is a lightweight helper. For more control and robustness use
    `scipy.signal.find_peaks`.

    Parameters
    - signal: 1-D sequence
    - min_height: minimum required prominence above local neighbors

    Returns: numpy array of integer indices
    """
    arr = np.asarray(signal)
    if arr.size < 3:
        return np.array([], dtype=int)

    diff_signal = np.diff(arr)
    candidate_peaks = np.where((diff_signal[:-1] > 0) & (diff_signal[1:] < 0))[0] + 1
    peaks = [int(i) for i in candidate_peaks if arr[i] - np.min(arr[max(0, i - 1):i + 2]) >= min_height]
    return np.array(peaks, dtype=int)


def find_valleys(signal: Sequence[float], min_depth: float = 0.0) -> np.ndarray:
    """Return sample indices of local minima using simple zero-crossing of derivative.

    Parameters
    - signal: 1-D sequence
    - min_depth: minimum required depth below local neighbors

    Returns: numpy array of integer indices
    """
    arr = np.asarray(signal)
    if arr.size < 3:
        return np.array([], dtype=int)
    diff_signal = np.diff(arr)
    candidate_valleys = np.where((diff_signal[:-1] < 0) & (diff_signal[1:] > 0))[0] + 1
    valleys = [int(i) for i in candidate_valleys if np.max(arr[max(0, i - 1):i + 2]) - arr[i] >= min_depth]
    return np.array(valleys, dtype=int)


def calculate_areas_trapz(frequencies: Sequence[float], power: Sequence[float], points: Sequence[int]) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Integrate `power` over frequency intervals defined by indices in `points`.

    Parameters
    - frequencies: 1-D frequency vector (Hz)
    - power: 1-D or 2-D power vector aligned with `frequencies` (if 2-D, integrates per column)
    - points: sequence of integer indices (monotonic) defining interval boundaries

    Returns
    - areas: list of trapezoidal integral values per interval
    - intervals: list of (f_start, f_end) tuples for each interval
    """
    freqs = np.asarray(frequencies)
    pw = np.asarray(power)
    pts = np.asarray(points, dtype=int)

    if pts.size < 2:
        return [], []

    areas: List[float] = []
    intervals: List[Tuple[float, float]] = []

    for i in range(pts.size - 1):
        start_idx = int(pts[i])
        end_idx = int(pts[i + 1])
        if start_idx < 0 or end_idx >= freqs.size or start_idx >= end_idx:
            continue

        freq_range = freqs[start_idx:end_idx + 1]
        power_range = pw[start_idx:end_idx + 1]

        area = float(np.trapz(power_range, x=freq_range))
        areas.append(area)
        intervals.append((float(freqs[start_idx]), float(freqs[end_idx])))
    return areas, intervals