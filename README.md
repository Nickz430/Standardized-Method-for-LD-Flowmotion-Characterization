# Laser Doppler Flowmotion - Signal Processing Pipeline

A comprehensive signal processing framework for analyzing Laser Doppler Flowmetry (LDF) signals to characterize microcirculatory dynamics and vasomotion/flowmotion phenomena in both healthy and critically ill populations.

## Overview

This pipeline implements a robust method to evaluate skin LDF signals captured during vascular occlusion challenges, enabling characterization of microvascular blood flow regulation. The analysis combines multiple signal processing techniques to extract physiologically meaningful features from complex LDF recordings.

### Key Features

- **Hampel Filtering**: Robust outlier detection and removal while preserving signal dynamics
- **Envelope Analysis**: Computation of flowmotion area through upper/lower signal envelopes
- **Continuous Wavelet Transform (CWT)**: Time-frequency decomposition of signals with physiological frequency band resolution
- **Power Spectral Density (PSD)**: Quantification of signal power across regulatory frequency bands
- **Automated Feature Extraction**: Comprehensive time and spectral domain characteristics

## Scientific Background

### Vasomotion and Flowmotion

Vasomotion refers to rhythmic oscillations in arteriole tone that produce periodic changes in vessel diameter and vascular resistance. These oscillations translate into significant variations in capillary perfusion and blood flow, collectively termed the flowmotion phenomenon. Preservation of vasomotion/flowmotion capacity and ability to restore tissue perfusion after transient ischemia reflect healthy microcirculation.

### Frequency Band Analysis

LDF signal decomposition reveals characteristic oscillatory components associated with specific regulatory mechanisms:

| Frequency Band | Range (Hz) | Regulatory Mechanism |
|---|---|---|
| Endothelial (NO-dependent) | 0.0095–0.021 | Endothelial function |
| Neurogenic | 0.021–0.052 | Sympathetic nervous system |
| Myogenic | 0.052–0.15 | Vascular smooth muscle |
| Respiratory | 0.15–0.4 | Breathing activity |
| Cardiac | 0.4–2.0 | Heart rate coupling |

### Vascular Occlusion Protocol

The analysis framework utilizes signals acquired during a standardized three-phase protocol:

1. **Pre-ischemia (Baseline)**: 3-minute baseline measurement
2. **Ischemia**: 3-minute transient brachial artery occlusion (50 mmHg above systolic pressure)
3. **Reperfusion**: 3-minute post-occlusion recovery period

## Pipeline Architecture

### Processing Steps

The complete analysis pipeline executes six sequential steps:

#### Step 1: Hampel Outlier Filtering
Robust filtering removes signal artifacts and outliers while preserving physiological oscillations.
- **Input**: Raw LDF signals
- **Output**: Filtered signals (data/filtered/)

#### Step 2: Signal Vasomotion Area Analysis
Computes envelope-based metrics characterizing flowmotion amplitude.
- **Input**: Filtered signals
- **Output**: Signal comparisons and vasomotion areas

#### Step 3: Spectrogram Generation (CWT)
Time-frequency analysis using Continuous Wavelet Transform with Complex Morlet wavelet (cmor1.5-1.0).
- **Input**: Filtered signals
- **Output**: Spectrograms of a Healthy volunteer and a septic shock patient
- **Frequency Range**: 0.001–2.5 Hz

#### Step 4: PSD Analysis
Computes power spectral density within physiological frequency bands.
- **Input**: CWT power matrices
- **Output**: Band-specific power analysis

#### Step 5: Power Spectrum Comparison
Overlay plots comparing healthy vs. septic patient spectral characteristics.
- **Input**: Average power spectra
- **Output**:  Comparative PSD between healthy subject and septic shock patient

## Installation

### Requirements

- Python 3.11+
- See `requirements.txt` for package dependencies

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nickz430/LD-Flowmotion.git
   cd LD-Flowmotion
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Complete Pipeline

Execute all analysis steps in sequence:

```bash
python "src/Signal Analysis/run_complete_analysis.py"
```

The pipeline will:
- Validate script availability
- Execute each step sequentially
- Provide detailed timing and status information
- Generate outputs in `plt/` and `Tables/` folders
- Halt on first error with diagnostic information


## Key References

This pipeline implements methods described in contemporary LDF analysis literature:

- Frequency band analysis in LDF signals
- Wavelet decomposition for physiological signal characterization
- Multiscale entropy techniques for nonlinear dynamics
- Vascular occlusion protocol for assessing autoregulation

## Repository Management

This repository uses a structured `.gitignore` that:
- Tracks all source code and data processing scripts
- Preserves input and filtered data folders
- Ignores generated outputs (figures, tables, processed files)
- Excludes Python cache, virtual environments, and IDE settings

To clone and work with this repository:

```bash
git clone https://github.com/Nickz430/LD-Flowmotion.git
```

## Troubleshooting

### Unicode Encoding Errors
The pipeline handles Windows console encoding limitations with ASCII-safe status markers (`[OK]`, `[ERR]`).

### Script Not Found Errors
The pipeline uses dynamic script resolution to accommodate flexible folder layouts. If a script isn't found in the expected location, the system searches the entire repository structure.

### Column Name Mismatches
Signal column detection is case-insensitive and supports multiple naming conventions (e.g., `healthy_subject`, `Sano`, `sano2`).

## Use and Citation

This project is free to download and use. 

If you use this software in your research, please cite using the following format:

```bibtex
Pending for publication
```

Or in plain text:

Laser Doppler Flowmotion signal analysis pipeline; Nicolas Orozco MD MSc; GitHub repository; Version 1.0.0; https://github.com/Nickz430/LD-Flowmotion

## Contact and Contributions

This repository is maintained by **Nicolas Orozco MD MSc** (nicolas.orozco.e@hotmail.com).

Contributions in the form of questions, suggestions, issues, and pull requests are welcome and encouraged.

For:
- **Bug reports**: Please open an issue with detailed description and reproducible steps
- **Feature requests**: Submit an issue describing the desired functionality
- **Code contributions**: Fork the repository, create a feature branch, and submit a pull request
- **Questions**: Open a discussion or contact via email

---

**Note**: This framework is designed for research applications analyzing Laser Doppler Flowmetry signals. Clinical interpretation requires appropriate medical expertise and validation within clinical contexts.
