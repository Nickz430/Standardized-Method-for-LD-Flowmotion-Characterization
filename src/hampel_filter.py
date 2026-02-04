import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
from typing import Dict, Optional, Tuple
from scipy.stats import median_abs_deviation

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", "r"))
signal_analysis_cfg = cfg["SIGNAL_ANALYSIS"]["DATASETS"]
path_cfg = signal_analysis_cfg["PATHS"]

## CONSTANTS
K = 10 # Window size 
T0 = 4 # Threshold multiplier for the MAD

def hampel_filter(
        x : pd.Series, 
        k : int = 10, 
        t0: int = 4) -> pd.Series:
    """
    Hampel filter to remove outliers.
    x: Input data series
    k: Window size (default: 10)
    t0: Threshold multiplier for the MAD (default: 4)
    """
    
    x = x.astype(float)
    n = len(x)
    y = x.fillna(0)
    L = 1.4826 
    
    for i in range(k, n - k):
        window_data = x[i - k:i + k + 1]
        median_val = np.median(window_data)
        MAD = L * median_abs_deviation(window_data, scale=1)
        
        if abs(x[i] - median_val) > t0 * MAD:
            y[i] = median_val  
    
    return y

def plot_filtered_data(
        signal : pd.Series,
        save_path: Optional[str] = None):
    """
    Function to plot original vs. filtered Laser Doppler data.
    """
    signal_filter = hampel_filter(signal, k=K, t0=T0)
    plt.figure(figsize=(10, 5))
    plt.plot(signal, color='red', label='Original')
    plt.plot(signal_filter, color='blue', label='Filtered')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

signal_rawdataset = pd.read_excel(os.path.join(path_cfg["INPUT_DIR"],path_cfg["RAW_DIR"], path_cfg["RAW_FILE"]))

signal_filtered = signal_rawdataset.copy()
for i in signal_filtered.columns[1:-1]:
    signal_filtered[i] = hampel_filter(signal_filtered[i],k=K, t0=T0)

signal_filtered.to_excel(os.path.join(path_cfg["OUTPUT_DIR"],path_cfg["FILTERED_FILE"]),index=False)
print(f"Signals filtered and saved in file {path_cfg['FILTERED_FILE']}")