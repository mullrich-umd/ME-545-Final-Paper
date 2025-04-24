# Vehicle Audio Spectrum Analysis Toolkit

This repository contains Python scripts designed to analyze audio spectrum data, primarily focused on vehicle interior noise measurements. It processes raw spectrum data (e.g., exported from Audacity), calculates various acoustic metrics like Sound Pressure Level (SPL), A-weighted SPL (dBA), 1/3 octave band levels, Articulation Index (AI), and Loudness (Sones), and generates a series of plots for analysis and comparison.

## Core Components

1.  **`audio_spl_lib.py`**: A library containing functions for:
    *   Reading spectrum data files.
    *   Calculating A-weighting.
    *   Calculating narrow-band and 1/3 octave band SPL (dBZ and dBA).
    *   Calculating Articulation Index (AI) based on ANSI S3.5.
    *   Calculating Loudness (Sones) using the MOSQITO library (Zwicker method, ISO 532-1).
    *   Generating various plots (narrow-band, 1/3 octave, AI details, summary plots).
    *   Helper functions (e.g., sorting labels).
2.  **`main_vehicle_analysis.py`**: The main application script that:
    *   Uses `audio_spl_lib.py` to process input files or directories.
    *   Manages gain offsets for different vehicles/setups via `gain_offsets.json`.
    *   Orchestrates the calculation of metrics for individual files and entire vehicles.
    *   Generates individual plots, combined plots per vehicle, summary plots (AI/Loudness vs. Setting), and comparison plots across vehicles.
3.  **`gain_offsets.json`**: (Created automatically) Stores the microphone/recording gain offset in dB for each vehicle to ensure accurate SPL calculations.

## Features

*   Reads tab-delimited spectrum data files (Frequency Hz \t Level dB).
*   Calculates narrow-band SPL (dBZ and dBA).
*   Calculates 1/3 octave band SPL (dBZ and dBA) based on ANSI S1.11 / IEC 61260 standard frequencies.
*   Calculates Articulation Index (AI) according to ANSI S3.5 methodology.
*   Calculates psychoacoustic Loudness (Sones) using the Zwicker method via the MOSQITO library (optional).
*   Generates individual plots for each measurement:
    *   Narrow-Band SPL Spectrum
    *   1/3 Octave Band SPL Spectrum
    *   Articulation Index Band Detail Plot
*   Generates combined narrow-band SPL plots showing all settings for a single vehicle.
*   Generates summary plots per vehicle:
    *   Articulation Index vs. Blower Setting
    *   Loudness vs. Blower Setting
*   Generates comparison plots across multiple vehicles:
    *   Grouped 1/3 Octave SPL by setting
    *   Overlaid Narrow-Band SPL by setting
    *   Grouped Articulation Index by setting
    *   Grouped Loudness by setting
*   Persistent storage of gain offsets.
*   Automatic extraction of vehicle names and settings from file/directory structure.
*   Organized output directory structure for generated plots (SVG format).

## Requirements

*   Python 3.x
*   NumPy
*   Matplotlib
*   **MOSQITO** (Optional, but required for Loudness calculations)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install required Python packages:**
    ```bash
    pip install numpy matplotlib
    ```
3.  **Install MOSQITO (Optional):**
    If you want to calculate Loudness (Sones), install the MOSQITO library:
    ```bash
    pip install mosqito
    ```
    If MOSQITO is not installed, the script will print a warning and skip loudness calculations.

## Usage

1.  **Prepare Input Data:**
    *   Place your spectrum data files (`.txt`) in directories named after the vehicles (e.g., `Vehicle Data/Ford Explorer/`, `Vehicle Data/Toyota Tundra/`).
    *   Ensure the text files are tab-delimited with two columns: Frequency (Hz) and Level (dB). Typically, this is the format exported by Audacity's "Spectrum Export" feature (using default settings, ensuring sufficient FFT points for resolution).
    *   Name the files descriptively, including the blower setting (e.g., `Explorer_Setting_1.txt`, `Tundra_Setting_Off.txt`). The script attempts to extract the setting ("Off", "1", "2", etc.) from the filename.
2.  **Configure Input Paths:**
    *   Edit the `INPUT_PATHS` list within `main_vehicle_analysis.py` to point to the directories (or specific files) you want to process.
    ```python
    # --- User Configuration ---
    INPUT_PATHS = [
        "Vehicle Data/Ford Explorer",
        "Vehicle Data/Honda Odyssey", # Corrected spelling?
        "Vehicle Data/Toyota Tundra"
    ]
    ```
3.  **Configure Options (Optional):**
    *   `APPLY_A_WEIGHTING_TO_PLOTS` (in `main_vehicle_analysis.py`): Set to `True` (default) to generate primary NB/Octave plots in dBA. Set to `False` for dBZ plots. Note: AI calculation *always* uses dBA internally, regardless of this flag. Loudness calculation *always* uses dBZ internally.
    *   Plotting constants (dimensions, axis limits, etc.) can be adjusted in `audio_spl_lib.py`.
4.  **Run the Script:**
    ```bash
    python main_vehicle_analysis.py
    ```
5.  **Enter Gain Offsets:**
    *   The first time you run the script for a new vehicle, it will prompt you to enter the gain offset (in dB) for that vehicle's recording setup. This value compensates for microphone sensitivity, preamp gain, etc., to achieve calibrated SPL values.
    *   These offsets are saved in `gain_offsets.json` in the same directory as the script, so you won't be prompted again for the same vehicle.
6.  **Review Output:**
    *   Plots will be saved as SVG files in subdirectories within each vehicle's input directory (`NarrowBand`, `ThirdOctave`, `Combined`, `ArticulationIndex`, `Loudness`).
    *   Comparison plots across all processed vehicles will be saved in a top-level `Comparison_Plots` directory created alongside the script.

## Input Data Format

*   **File Type:** Plain text (`.txt`)
*   **Delimiter:** Tab (`\t`)
*   **Columns:**
    1.  Frequency (Hz)
    2.  Level (dB) - Raw level from the spectrum analyzer (e.g., Audacity)
*   **Header:** The script assumes **1 header row** to skip (configurable via `HEADER_ROWS_TO_SKIP` in `audio_spl_lib.py`).
*   **Low Levels:** Values below `AUDACITY_LOW_LEVEL_THRESHOLD` (e.g., -999 dB) are treated as negative infinity.

**Example (Audacity Spectrum Export):**
Make sure Audacity is configured to export spectrum data with appropriate FFT size (e.g., 8192, 16384, or higher for good low-frequency resolution) and windowing function (e.g., Hanning).

## Key Calculations

*   **SPL (Sound Pressure Level):** Calculated from the input dB levels, converting amplitude to RMS pressure, referencing \( P_{ref} = 20 \mu Pa \), and applying the user-provided gain offset. Both unweighted (dBZ) and A-weighted (dBA) values are calculated.
*   **A-Weighting:** Applied according to the formula in IEC 61672-1:2013.
*   **1/3 Octave Bands:** Calculated by summing the power (derived from narrow-band SPL) within the frequency limits defined by ANSI S1.11 / IEC 61260 for standard nominal center frequencies.
*   **Articulation Index (AI):** Calculated based on ANSI S3.5. It uses the A-weighted 1/3 octave band levels between 200 Hz and 6300 Hz and applies specific weighting factors to determine a percentage score (0-100%) representing speech intelligibility in the presence of noise.
*   **Loudness (Sones):** Calculated using the `loudness_zwst_freq` function from the MOSQITO library, which implements the Zwicker Loudness model (ISO 532-1) for stationary sounds based on a free-field assumption. It requires the unweighted narrow-band SPL spectrum (converted to pressure) as input.