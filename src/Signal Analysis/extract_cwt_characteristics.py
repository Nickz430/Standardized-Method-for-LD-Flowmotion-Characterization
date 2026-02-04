"""
Extract comprehensive signal characteristics from healthy and septic signals.

This module processes two representative signals (one healthy and one septic shock patient)
and computes time-domain and wavelet-based spectral features using Continuous Wavelet 
Transform (CWT) analysis. Results are saved to both Excel and text file formats.

Features Extracted:
    - Time Domain: Mean, Variance, Standard Deviation, Energy
    - Spectral Domain: Maximum Peak Frequency, Maximum Power, Entropy
    - Frequency Band Density: Power distribution across physiological frequency bands
    - Signal Characteristics: Number of peaks and valleys in the power spectrum

Author: Signal Processing Pipeline
Date: 2026
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

from signal_processing_functions import (
    cwt_spectrogram, 
    find_peaks, 
    find_valleys, 
    calculate_areas_trapz
)


def extract_signal_characteristics(signal: np.ndarray, frequencies: np.ndarray, 
                                    avg_power: np.ndarray, signal_name: str) -> dict:
    """
    Extract comprehensive characteristics from a single signal.
    
    Parameters
    ----------
    signal : np.ndarray
        The raw time-domain signal.
    frequencies : np.ndarray
        Frequency array from CWT analysis.
    avg_power : np.ndarray
        Average power spectrum from CWT.
    signal_name : str
        Name/identifier for the signal (e.g., 'Healthy Subject', 'Septic Patient').
    
    Returns
    -------
    dict
        Dictionary containing all extracted characteristics with descriptive keys.
    """
    characteristics = {}
    
    # Time Domain Characteristics
    characteristics['Signal_Name'] = signal_name
    characteristics['Mean_Signal'] = np.mean(signal)
    characteristics['Variance_Signal'] = np.var(signal)
    characteristics['Std_Dev_Signal'] = np.std(signal)
    characteristics['Energy_Signal'] = np.sum(np.square(signal)) / len(signal)
    
    # Spectral Features - Maximum Power
    peak_index = np.argmax(avg_power)
    characteristics['MaxPeak_Frequency_Hz'] = frequencies[peak_index]
    characteristics['MaxPeak_Power'] = avg_power[peak_index]
    
    # Spectral Entropy
    # Normalize power spectrum to create a probability distribution
    normalized_power = avg_power / np.sum(avg_power)
    # Calculate entropy (avoid log(0) by adding small epsilon)
    entropy = -np.sum(normalized_power * np.log2(normalized_power + np.finfo(float).eps))
    characteristics['Spectral_Entropy'] = entropy
    
    # Peak and Valley Detection
    peaks = find_peaks(avg_power)
    valleys = find_valleys(avg_power)
    characteristics['Number_of_Peaks'] = len(peaks)
    characteristics['Number_of_Valleys'] = len(valleys)
    
    # Spectral Density in Frequency Bands
    # Define frequency boundaries for physiological frequency bands
    # [Endothelial, Neurogenic, Myogenic, Respiratory, Cardiac, upper limit]
    band_boundaries = [0, 0.02, 0.06, 0.15, 0.4, 2.5]
    band_names = ['Endothelial', 'Neurogenic', 'Myogenic', 'Respiratory', 'Cardiac']
    
    # Find indices corresponding to frequency boundaries
    band_indices = [np.argmin(np.abs(frequencies - freq)) for freq in band_boundaries]
    
    # Calculate PSD in each band
    areas_spectrum, _ = calculate_areas_trapz(frequencies, avg_power, band_indices)
    
    for i, band_name in enumerate(band_names):
        characteristics[f'PSD_{band_name}_Band'] = areas_spectrum[i]
    
    return characteristics


def main():
    """
    Main execution function.
    
    Workflow:
        1. Load filtered signal data from Excel file
        2. Extract two representative signals (healthy and septic)
        3. Compute Continuous Wavelet Transform for each signal
        4. Extract time and spectral domain characteristics
        5. Save results to Excel and formatted text file
    """
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_file = repo_root / 'data' / 'filtered' / 'signals_filtered.xlsx'

    if not data_file.exists():
        raise FileNotFoundError(f'Data file not found at: {data_file}')

    print("=" * 80)
    print("Signal Characteristics Extraction - Two Representative Subjects")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    df = pd.read_excel(data_file)
    
    # Filter data: keep only reperfusion phase (Time > 213.7 seconds)
    data_reperfusion = df[df['Time'] > 213.7].reset_index(drop=True)
    print(f"Data shape after filtering: {data_reperfusion.shape}")

    # Robust column selection: allow several common column name variants
    def _pick_column(df_local, candidates):
        cols = list(df_local.columns)
        cols_lower = [c.lower() for c in cols]
        # 1) exact match on candidate names
        for cand in candidates:
            if cand in cols:
                return cand
        # 2) case-insensitive exact
        for cand in candidates:
            for col, col_l in zip(cols, cols_lower):
                if col_l == cand.lower():
                    return col
        # 3) substring match (e.g., 'sano' in 'Sano2')
        for cand in candidates:
            cand_l = cand.lower()
            for col, col_l in zip(cols, cols_lower):
                if cand_l in col_l:
                    return col
        raise KeyError(f"None of the candidate columns found in dataframe: {candidates}")
    
    # Define frequency intervals for Continuous Wavelet Transform
    frequency_intervals1 = np.arange(0.001, 0.5, 0.001)
    frequency_intervals2 = np.arange(0.501, 2.5, 0.01)
    frequency_intervals = np.concatenate((frequency_intervals1, frequency_intervals2))
    
    # Parameters
    sampling_frequency = 10  # Hz
    results = []
    
    # === Process Two Representative Signals ===
    print("\n" + "-" * 80)
    print("Processing signals...")
    print("-" * 80)
    
    # Identify the actual column names for healthy and septic signals
    healthy_candidates = ['healthy_subject', 'healthy subject', 'sano', 'sano2', 'sano_2']
    sepsis_candidates = ['septic_patient', 'septic patient', 'sepsis', 'septic']

    try:
        healthy_col = _pick_column(data_reperfusion, healthy_candidates)
        sepsis_col = _pick_column(data_reperfusion, sepsis_candidates)
    except KeyError as e:
        print(f"[ERR] Column detection error: {e}")
        print("Available columns:", list(data_reperfusion.columns))
        return

    signals_to_process = {
        healthy_col: 'Healthy Subject',
        sepsis_col: 'Septic Shock Patient'
    }

    for signal_column, signal_name in signals_to_process.items():
        try:
            # Extract signal
            signal = data_reperfusion[signal_column].dropna().values
            if len(signal) == 0:
                print(f"  [ERR] {signal_name}: No data available")
                continue
            
            print(f"\n  Processing {signal_name}...")
            print(f"    - Signal length: {len(signal)} samples")
            print(f"    - Duration: {len(signal)/sampling_frequency:.1f} seconds")
            
            # Compute Continuous Wavelet Transform
            t0 = time.time()
            power, times, frequencies, coif = cwt_spectrogram(
                signal, 
                sampling_frequency, 
                frequency_intervals,
                signal_detrend=True, 
                normalize=True
            )
            cwt_time = time.time() - t0
            print(f"    - CWT computed in {cwt_time:.2f}s")
            
            # Extract average power spectrum
            freq_filter = (frequencies >= 0.001) & (frequencies <= 2.5)
            avg_power = np.mean(power[freq_filter, :], axis=1)
            
            # Extract characteristics
            characteristics = extract_signal_characteristics(
                signal, 
                frequencies[freq_filter], 
                avg_power,
                signal_name
            )
            results.append(characteristics)
            print(f"    [OK] {len(characteristics)} characteristics extracted")
            
        except KeyError:
            print(f"  [ERR] {signal_name}: Signal column not found in data")
            continue
        except Exception as e:
            print(f"  [ERR] {signal_name}: Error during processing - {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nSignals processed: {len(results_df)}")
    print(f"Characteristics extracted per signal: {len(results_df.columns)}")
    
    # === Save Results ===
    # Excel file
    excel_file = repo_root / 'Tables' / 'signal_characteristics.xlsx'
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(excel_file, index=False)
    print(f"\n[OK] Excel file saved: {excel_file}")
    
    # Text file with detailed report
    txt_file = repo_root / 'Tables' / 'signal_characteristics.txt'
    with open(txt_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("SIGNAL CHARACTERISTICS EXTRACTION REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        # Header information
        f.write("ANALYSIS PARAMETERS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sampling Frequency: {sampling_frequency} Hz\n")
        f.write(f"Time Window: Reperfusion phase (Time > 213.7 seconds)\n")
        f.write(f"Frequency Range: 0.001 - 2.5 Hz\n")
        f.write(f"Transform Method: Continuous Wavelet Transform (CWT)\n")
        f.write(f"Wavelet: Complex Morlet (cmor1.5-1.0)\n\n")
        
        # Results table
        f.write("=" * 100 + "\n")
        f.write("EXTRACTED CHARACTERISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        # Transpose for better readability (characteristics as rows, signals as columns)
        results_transposed = results_df.set_index('Signal_Name').T
        f.write(results_transposed.to_string())
        f.write("\n\n")
        
        # Feature descriptions
        f.write("=" * 100 + "\n")
        f.write("FEATURE DESCRIPTIONS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("TIME DOMAIN CHARACTERISTICS:\n")
        f.write("-" * 100 + "\n")
        f.write("  Mean_Signal\n")
        f.write("      Average amplitude of the raw signal.\n")
        f.write("      Unit: Blood Perfusion Units (BPU)\n\n")
        f.write("  Variance_Signal\n")
        f.write("      Measure of signal variability around the mean.\n")
        f.write("      Higher values indicate greater fluctuations.\n\n")
        f.write("  Std_Dev_Signal\n")
        f.write("      Standard deviation of the raw signal.\n")
        f.write("      Square root of variance; same units as signal.\n\n")
        f.write("  Energy_Signal\n")
        f.write("      Mean squared value of the signal.\n")
        f.write("      Represents total signal power in time domain.\n\n")
        
        f.write("SPECTRAL FEATURES (CWT-BASED):\n")
        f.write("-" * 100 + "\n")
        f.write("  MaxPeak_Frequency_Hz\n")
        f.write("      Frequency at which maximum power occurs in the spectrum.\n")
        f.write("      Indicates the dominant oscillation frequency.\n\n")
        f.write("  MaxPeak_Power\n")
        f.write("      Maximum power value in the frequency spectrum.\n")
        f.write("      Represents the strength of the dominant frequency component.\n\n")
        f.write("  Spectral_Entropy\n")
        f.write("      Measure of disorder/randomness in the frequency spectrum.\n")
        f.write("      Low entropy: narrow frequency bands (organized signal)\n")
        f.write("      High entropy: distributed frequency energy (random signal)\n\n")
        
        f.write("PEAK AND VALLEY DETECTION:\n")
        f.write("-" * 100 + "\n")
        f.write("  Number_of_Peaks\n")
        f.write("      Count of local maxima in the power spectrum.\n")
        f.write("      Indicates number of distinct oscillation frequencies.\n\n")
        f.write("  Number_of_Valleys\n")
        f.write("      Count of local minima in the power spectrum.\n")
        f.write("      Indicates frequency ranges with low activity.\n\n")
        
        f.write("PHYSIOLOGICAL FREQUENCY BANDS:\n")
        f.write("-" * 100 + "\n")
        f.write("  PSD_Endothelial_Band (0.0095 - 0.02 Hz)\n")
        f.write("      PSD in the endothelial (NO-mediated) regulation band.\n")
        f.write("      Reflects endothelial function.\n\n")
        f.write("  PSD_Neurogenic_Band (0.02 - 0.06 Hz)\n")
        f.write("      PSD in the neurogenic (sympathetic) regulation band.\n")
        f.write("      Reflects sympathetic nervous system activity.\n\n")
        f.write("  PSD_Myogenic_Band (0.06 - 0.15 Hz)\n")
        f.write("      PSD in the myogenic (vascular smooth muscle) band.\n")
        f.write("      Reflects vascular smooth muscle oscillations.\n\n")
        f.write("  PSD_Respiratory_Band (0.15 - 0.4 Hz)\n")
        f.write("      PSD in the respiratory frequency band.\n")
        f.write("      Coupled to breathing activity.\n\n")
        f.write("  PSD_Cardiac_Band (0.4 - 2.5 Hz)\n")
        f.write("      PSD in the cardiac frequency band.\n")
        f.write("      Coupled to heart rate.\n\n")
        

    print(f"[OK] Text report saved: {txt_file}\n")
    print("=" * 80)
    print("Analysis complete!\n")


if __name__ == '__main__':
    main()
