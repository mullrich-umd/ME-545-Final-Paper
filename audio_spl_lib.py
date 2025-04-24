# audio_spl_lib.py
# Library for basic audio spectrum processing and plotting

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # Added for NullLocator
import math
import os
import re

# Attempt to import MOSQITO loudness function
try:
    from mosqito.sq_metrics import loudness_zwst_freq
    MOSQITO_AVAILABLE = True
except ImportError:
    print("Warning: MOSQITO library not found. Loudness calculations will be skipped.")
    print("         Install it using: pip install mosqito")
    MOSQITO_AVAILABLE = False
    # Define a dummy function if MOSQITO is not available
    def loudness_zwst_freq(*args, **kwargs):
        print("Error: MOSQITO not available, cannot calculate loudness.")
        # Return dummy values matching the expected output structure
        return np.nan, np.array([np.nan]), np.array([np.nan])


# --- Calculation Constants ---
P_REF = 0.00002 # Reference pressure for SPL (Pascal)
SMALL_EPSILON = 1e-12 # Small value to avoid log10(0) or division by zero
A_WEIGHTING_EPSILON = 1e-16 # Specific epsilon for A-weighting calculation stability
DB_AMPLITUDE_FACTOR = 20.0 # Factor for dB calculation from amplitude/pressure
DB_POWER_FACTOR = 10.0 # Factor for dB calculation from power

# --- File Reading Constants ---
HEADER_ROWS_TO_SKIP = 1 # Number of header rows in Audacity spectrum files
MIN_SPECTRUM_COLUMNS = 2 # Expect at least Frequency and Level columns
AUDACITY_LOW_LEVEL_THRESHOLD = -999.0 # dB values below this are treated as -inf

# --- Default Plotting Constants ---
# These can potentially be overridden by passing arguments to plot functions
PLOT_FREQ_MIN_HZ = 100
PLOT_FREQ_MAX_HZ = 10000
PLOT_NB_Y_MIN_DB = -20 # Fixed Y-axis min for narrow-band plots
PLOT_NB_Y_MAX_DB = 50  # Fixed Y-axis max for narrow-band plots
PLOT_OCTAVE_Y_MIN_DB = 5   # Fixed Y-axis min for 1/3 octave plots
PLOT_OCTAVE_Y_MAX_DB = 55  # Fixed Y-axis max for 1/3 octave plots
PLOT_AI_DETAIL_Y_MIN_DB = 0 # Fixed Y-axis min for AI detail plots
PLOT_AI_DETAIL_Y_MAX_DB = 80 # Fixed Y-axis max for AI detail plots
PLOT_LOUDNESS_Y_MIN = 0 # Fixed Y-axis min for Loudness plots
PLOT_LINEWIDTH = 1.0
PLOT_GRID_LINEWIDTH = 0.5
PLOT_BAR_LINEWIDTH = 0.5 # Linewidth for octave bar edges and vlines
PLOT_XTICK_ROTATION = 45
PLOT_GRID_ALPHA = 0.7

PLOT_AI_COMP_Y_MIN = 0   # Fixed Y-axis min for AI Comparison plots
PLOT_AI_COMP_Y_MAX = 105 # Fixed Y-axis max for AI Comparison plots (AI is 0-100)

# Get default color cycle for consistent vehicle colors
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# --- Standard Frequencies ---
# Define the nominal 1/3 octave center frequencies (Hz) you care about
# (ANSI S1.11-2004 / IEC 61260-1:2014 standard frequencies)
NOMINAL_FREQUENCIES = np.array([
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
    5000, 6300, 8000, 10000, 12500, 16000, 20000
])

AI_PARAMETERS = {
    # Freq: (Lower dB(A), Upper dB(A), Weighting Factor)
    200:  (23.1, 53.1, 1.00),
    250:  (30.4, 60.4, 2.00),
    315:  (34.4, 64.4, 3.25),
    400:  (38.2, 68.2, 4.25),
    500:  (41.8, 71.8, 4.50),
    630:  (43.1, 73.1, 5.25),
    800:  (44.2, 74.2, 6.50),
    1000: (44.0, 74.0, 7.25),
    1250: (42.6, 72.6, 8.50),
    1600: (41.0, 71.0, 11.50),
    2000: (38.2, 68.2, 11.00),
    2500: (36.3, 66.3, 9.50),
    3150: (34.2, 64.2, 9.00),
    4000: (31.0, 61.0, 7.75),
    5000: (26.5, 56.5, 6.25),
    6300: (20.9, 50.9, 2.50),
}
AI_FREQUENCIES = np.array(list(AI_PARAMETERS.keys()))

# --- Calculation Functions ---

def calculate_a_weighting(freq):
    """
    Calculates the A-weighting correction in dB for given frequencies.
    Uses the formula from IEC 61672-1:2013. Handles f=0 and potential numerical issues.

    Args:
        freq (np.array or float): Frequency or array of frequencies in Hz.

    Returns:
        np.array or float: A-weighting correction(s) in dB. Returns -inf for freq <= 0.
    """
    freq = np.asarray(freq, dtype=float) # Ensure input is a numpy float array
    result = np.full_like(freq, -np.inf) # Initialize with -inf

    # Process only positive frequencies
    pos_indices = freq > 0
    if not np.any(pos_indices):
        return result # Return -inf if no positive frequencies

    freq_pos = freq[pos_indices]
    f_sq = freq_pos**2

    # Constants for A-weighting formula (standard values)
    k1 = 20.6**2
    k2 = 107.7**2
    k3 = 737.9**2
    k4 = 12194**2
    C = 2.00 # Normalization constant in dB (approx. +1.9997 dB)

    # Calculate numerator and denominator parts
    num = k4 * (f_sq**2)
    # Use np.maximum to avoid issues with exactly zero frequency if it slips through
    den1 = np.maximum(f_sq + k1, A_WEIGHTING_EPSILON)
    den2 = np.maximum(f_sq + k2, A_WEIGHTING_EPSILON)
    den3 = np.maximum(f_sq + k3, A_WEIGHTING_EPSILON)
    den4 = np.maximum(f_sq + k4, A_WEIGHTING_EPSILON)

    # Calculate R_A(f)
    R_A = num / (den1 * np.sqrt(den2) * np.sqrt(den3) * den4)

    # Calculate A-weighting in dB
    # Add epsilon inside log10 for stability if R_A is extremely close to 0
    A_f = DB_AMPLITUDE_FACTOR * np.log10(R_A + A_WEIGHTING_EPSILON) + C

    result[pos_indices] = A_f

    # Ensure values are finite, replace NaNs or Infs resulting from edge cases
    result[~np.isfinite(result)] = -np.inf

    return result


def calculate_narrowband_spl(audacity_db, gain_offset, frequencies, apply_a_weighting=True):
    """
    Calculates narrow-band SPL from Audacity dB values.
    Always calculates unweighted (dBZ) SPL internally.
    Optionally applies A-weighting to produce dBA SPL.

    Args:
        audacity_db (np.array): Narrow-band levels from Audacity (can contain -inf).
        gain_offset (float): Gain offset in dB to apply.
        frequencies (np.array): Corresponding narrow-band frequencies.
        apply_a_weighting (bool): If True, the primary return SPL is A-weighted (dBA).
                                  If False, the primary return SPL is unweighted (dBZ).

    Returns:
        tuple: (
            spl_final (np.array): Calculated narrow-band SPL (dBA or dBZ based on flag), can contain -inf.
            spl_unweighted (np.array): Calculated unweighted narrow-band SPL (dBZ), can contain -inf.
        )
    """
    # Initialize SPL arrays with -inf
    spl_unweighted = np.full_like(audacity_db, -np.inf, dtype=float)
    spl_final = np.full_like(audacity_db, -np.inf, dtype=float)

    # Find indices where Audacity data is valid (not -inf)
    valid_audacity_indices = np.where(audacity_db > -np.inf)[0]

    if len(valid_audacity_indices) > 0:
        audacity_db_valid = audacity_db[valid_audacity_indices]
        frequencies_valid = frequencies[valid_audacity_indices]

        # Calculate pressure term only for valid indices
        amplitude_term = 10**(audacity_db_valid / DB_AMPLITUDE_FACTOR)
        pressure_term = amplitude_term / np.sqrt(2) # RMS conversion

        # Calculate unweighted SPL (dBZ), avoid log10(0) or log10(negative)
        pressure_ratio = np.maximum(pressure_term / P_REF, SMALL_EPSILON)
        spl_unweighted_valid = DB_AMPLITUDE_FACTOR * np.log10(pressure_ratio) + gain_offset

        # Place the calculated unweighted SPLs into the main spl_unweighted array
        spl_unweighted[valid_audacity_indices] = spl_unweighted_valid

        # Copy unweighted to final initially
        spl_final[valid_audacity_indices] = spl_unweighted_valid

        # Apply A-weighting to spl_final if requested
        if apply_a_weighting:
            # Find indices where the *calculated unweighted SPL* is finite
            finite_spl_indices_rel = np.where(spl_unweighted_valid > -np.inf)[0]

            if len(finite_spl_indices_rel) > 0:
                # Get the absolute indices corresponding to these finite SPLs
                finite_spl_indices_abs = valid_audacity_indices[finite_spl_indices_rel]
                freqs_for_weighting = frequencies[finite_spl_indices_abs]
                a_weights = calculate_a_weighting(freqs_for_weighting)

                # Find where the A-weights themselves are valid
                valid_weights_indices_rel = np.where(a_weights > -np.inf)[0]

                if len(valid_weights_indices_rel) > 0:
                    # Get absolute indices where both SPL and A-weight are valid
                    indices_to_update = finite_spl_indices_abs[valid_weights_indices_rel]
                    # Add the valid A-weights to the corresponding SPL values in spl_final
                    spl_final[indices_to_update] += a_weights[valid_weights_indices_rel]

    return spl_final, spl_unweighted


def spl_dbz_to_rms_pressure(spl_dbz, frequencies):
    """
    Converts unweighted narrow-band SPL (dBZ) values to RMS pressure (Pascals).

    Args:
        spl_dbz (np.array): Unweighted narrow-band SPL values (dBZ). Can contain -inf.
        frequencies (np.array): Corresponding frequencies (used for filtering invalid SPLs).

    Returns:
        np.array: RMS pressure values in Pascals. Values corresponding to
                  -inf dBZ or non-positive frequencies are set to 0.0.
    """
    rms_pressure = np.zeros_like(spl_dbz, dtype=float)

    # Find indices where SPL is finite and frequency is positive
    valid_indices = np.where((spl_dbz > -np.inf) & (frequencies > 0))[0]

    if len(valid_indices) > 0:
        spl_dbz_valid = spl_dbz[valid_indices]
        # Calculate RMS pressure using the inverse SPL formula
        pressure_valid = P_REF * (10**(spl_dbz_valid / DB_AMPLITUDE_FACTOR))
        rms_pressure[valid_indices] = pressure_valid

    return rms_pressure


def get_third_octave_bands(center_freqs):
    """
    Calculates lower and upper frequency limits for 1/3 octave bands.

    Args:
        center_freqs (np.array): Array of center frequencies.

    Returns:
        tuple: (center_freqs, lower_limits, upper_limits)
    """
    factor = 2**(1/6) # Standard factor for 1/3 octave
    lower_limits = center_freqs / factor
    upper_limits = center_freqs * factor
    return center_freqs, lower_limits, upper_limits

def calculate_third_octave_spl(nb_freqs, nb_spl, band_defs):
    """
    Calculates 1/3 octave band SPL by summing power from narrow-band SPL data.
    Works with either dBA or dBZ input for nb_spl.

    Args:
        nb_freqs (np.array): Narrow-band frequencies.
        nb_spl (np.array): Narrow-band SPL values (dBA or dBZ, can contain -inf).
        band_defs (tuple): (center_freqs, lower_limits, upper_limits)

    Returns:
        np.array: SPL (dBA or dBZ, matching input) for each 1/3 octave band (can contain -inf).
    """
    center_freqs, lower_limits, upper_limits = band_defs
    octave_spl_levels = np.full(len(center_freqs), -np.inf) # Initialize with -inf

    # Convert valid SPL values to linear power equivalents
    linear_power_equiv = np.zeros_like(nb_spl)
    valid_indices_spl = nb_spl > -np.inf
    if np.any(valid_indices_spl):
        spl_valid = nb_spl[valid_indices_spl]
        # Use DB_POWER_FACTOR for power summation
        linear_power_equiv[valid_indices_spl] = 10**(spl_valid / DB_POWER_FACTOR)

    for i in range(len(center_freqs)):
        f_lower = lower_limits[i]
        f_upper = upper_limits[i]
        # Find indices of narrow-band frequencies within the current 1/3 octave band
        indices_in_band = np.where((nb_freqs >= f_lower) & (nb_freqs < f_upper))[0]

        if len(indices_in_band) > 0:
            # Sum the linear power equivalents ONLY for the valid frequencies in this band
            valid_indices_in_band = np.intersect1d(indices_in_band, np.where(valid_indices_spl)[0])
            if len(valid_indices_in_band) > 0:
                total_power_in_band = np.sum(linear_power_equiv[valid_indices_in_band])

                # Convert total power back to dB, avoid log10(0)
                if total_power_in_band > SMALL_EPSILON:
                     # Use DB_POWER_FACTOR for conversion back to dB
                     octave_spl_levels[i] = DB_POWER_FACTOR * math.log10(total_power_in_band)

    return octave_spl_levels

def calculate_articulation_index(octave_center_freqs, octave_spl_a):
    """
    Calculates the Articulation Index (AI) based on ANSI S3.5.
    Requires A-weighted 1/3 octave SPL.

    Args:
        octave_center_freqs (np.array): Array of 1/3 octave center frequencies
                                         corresponding to the SPL values.
        octave_spl_a (np.array): Array of A-weighted 1/3 octave band SPL values (dBA).
                                 Must correspond to octave_center_freqs.
                                 Can contain -np.inf.

    Returns:
        float: Articulation Index value (0.0 to 100.0). Returns 0.0 if input
               data is invalid or no relevant bands are found.
    """
    ai_details = get_ai_calculation_details(octave_center_freqs, octave_spl_a)
    return ai_details['total_ai']


def get_ai_calculation_details(octave_center_freqs, octave_spl_a):
    """
    Performs the Articulation Index calculation and returns detailed results.
    Requires A-weighted 1/3 octave SPL.

    Args:
        octave_center_freqs (np.array): Array of 1/3 octave center frequencies.
        octave_spl_a (np.array): Array of A-weighted 1/3 octave band SPL values (dBA).

    Returns:
        dict: A dictionary containing:
            'total_ai': The final AI score (0-100).
            'contributions': {freq: contribution_value} for each AI band.
            'uncovered_fractions': {freq: uncovered_fraction} for each AI band.
            'spl_values': {freq: spl_value} for each AI band.
            'lower_limits': {freq: lower_limit} for each AI band.
            'upper_limits': {freq: upper_limit} for each AI band.
            'weights': {freq: weight} for each AI band.
    """
    total_weighted_contribution = 0.0
    window_width = 30.0 # Standard 30 dB window for AI bands
    results = {
        'total_ai': 0.0,
        'contributions': {},
        'uncovered_fractions': {},
        'spl_values': {},
        'lower_limits': {},
        'upper_limits': {},
        'weights': {}
    }

    # Create a mapping from nominal frequency to its index for quick lookup
    freq_to_index = {freq: i for i, freq in enumerate(octave_center_freqs)}

    for freq in AI_FREQUENCIES:
        if freq in AI_PARAMETERS and freq in freq_to_index:
            lower_limit, upper_limit, weight = AI_PARAMETERS[freq]
            spl_index = freq_to_index[freq]

            spl_val = -np.inf # Default if index is out of bounds or data missing
            if spl_index < len(octave_spl_a):
                # Ensure we use a finite value, default to -inf if NaN or Inf
                spl_val_raw = octave_spl_a[spl_index]
                if np.isfinite(spl_val_raw):
                    spl_val = spl_val_raw
                else:
                    spl_val = -np.inf # Treat non-finite input as infinitely low

            # Store details
            results['spl_values'][freq] = spl_val
            results['lower_limits'][freq] = lower_limit
            results['upper_limits'][freq] = upper_limit
            results['weights'][freq] = weight

            # Calculate uncovered fraction
            if spl_val == -np.inf or spl_val <= lower_limit:
                uncovered_fraction = 1.0 # Fully uncovered
            elif spl_val >= upper_limit:
                uncovered_fraction = 0.0 # Fully covered
            else:
                # Linear interpolation within the 30dB window
                uncovered_fraction = (upper_limit - spl_val) / window_width
                uncovered_fraction = max(0.0, min(1.0, uncovered_fraction)) # Clamp

            contribution = uncovered_fraction * weight
            results['uncovered_fractions'][freq] = uncovered_fraction
            results['contributions'][freq] = contribution
            total_weighted_contribution += contribution

        else:
             print(f"Warning: SPL data or AI parameters missing for AI frequency {freq} Hz.")
             # Store NaN or defaults for missing data if needed for plotting consistency
             results['spl_values'][freq] = np.nan
             results['lower_limits'][freq] = np.nan
             results['upper_limits'][freq] = np.nan
             results['weights'][freq] = 0.0
             results['uncovered_fractions'][freq] = 0.0
             results['contributions'][freq] = 0.0


    # Clamp final AI result
    results['total_ai'] = max(0.0, min(110.0, total_weighted_contribution)) # Allow slightly > 100
    return results


# --- File Reading Function ---

def read_spectrum_data(filepath):
    """
    Reads narrow-band spectrum data from a tab-delimited text file
    (expected format: Frequency\tLevel(dB)).

    Args:
        filepath (str): Path to the input text file.

    Returns:
        tuple: (frequencies (np.array), audacity_levels_db (np.array)) or (None, None) if error.
               Levels below AUDACITY_LOW_LEVEL_THRESHOLD are converted to -np.inf.
    """
    try:
        data = np.loadtxt(filepath, delimiter='\t', skiprows=HEADER_ROWS_TO_SKIP)
        if data.ndim != 2 or data.shape[1] < MIN_SPECTRUM_COLUMNS:
            print(f"Error: Data in {filepath} does not have at least {MIN_SPECTRUM_COLUMNS} columns.")
            return None, None
        frequencies = data[:, 0]
        audacity_levels_db = data[:, 1]
        # Replace extremely low values
        audacity_levels_db[audacity_levels_db < AUDACITY_LOW_LEVEL_THRESHOLD] = -np.inf
        # Ensure frequencies are positive for calculations
        if np.any(frequencies <= 0):
            print(f"Warning: File {filepath} contains non-positive frequencies. These will be ignored in calculations.")
            # Optionally filter them out here, or let downstream functions handle it
            # valid_freq_indices = frequencies > 0
            # frequencies = frequencies[valid_freq_indices]
            # audacity_levels_db = audacity_levels_db[valid_freq_indices]
        return frequencies, audacity_levels_db
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None, None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None

# --- Helper Functions ---

def get_setting_sort_key(label):
    """
    Generates a sort key for setting labels (e.g., "Off", "1", "4").
    "Off" is treated as the lowest setting (-1). Numeric settings are sorted numerically.
    Other strings are sorted last alphabetically.

    Args:
        label (str or int): The setting label.

    Returns:
        tuple: A tuple suitable for sorting (e.g., (-1, "OFF"), (0, 1), (1, "High")).
    """
    label_str = str(label).strip().upper()
    if label_str == "OFF":
        return (-1, "OFF") # Sorts first
    try:
        num = int(label_str)
        return (0, num) # Sort numerically
    except ValueError:
        try:
            num_float = float(label_str)
            return (0, num_float)
        except ValueError:
            return (1, label_str) # Sort other strings last

# --- Plotting Functions ---

def plot_narrowband_spl(frequencies, spl_levels, title, output_filename, width, height, weighted, color=None,
                        freq_min=PLOT_FREQ_MIN_HZ, freq_max=PLOT_FREQ_MAX_HZ,
                        y_min=PLOT_NB_Y_MIN_DB, y_max=PLOT_NB_Y_MAX_DB):
    """
    Creates and saves a narrow-band SPL spectrum plot.

    Args:
        frequencies (np.array): Frequency data.
        spl_levels (np.array): SPL data (dBA or dBZ, can contain -inf).
        title (str): Plot title.
        output_filename (str): Full path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        weighted (bool): If True, label y-axis as dBA, else dBZ.
        color (str, optional): Specific color for the plot line. Defaults to None (matplotlib default).
        freq_min (float, optional): Minimum frequency for plot x-axis. Defaults to PLOT_FREQ_MIN_HZ.
        freq_max (float, optional): Maximum frequency for plot x-axis. Defaults to PLOT_FREQ_MAX_HZ.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_NB_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_NB_Y_MAX_DB.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    # Plot only finite SPL values within the frequency range
    plot_indices = np.where((frequencies >= freq_min) & (frequencies <= freq_max) & np.isfinite(spl_levels))[0]

    if len(plot_indices) == 0:
        print(f"Warning: No finite data points between {freq_min} Hz and {freq_max} Hz to plot for {title}")
        plt.close(fig)
        return

    plot_freqs = frequencies[plot_indices]
    plot_spl = spl_levels[plot_indices]
    plot_color = color if color else None

    ax.plot(plot_freqs, plot_spl, linewidth=PLOT_LINEWIDTH, color=plot_color)
    ax.set_xscale('log')
    ax.set_xlabel("Frequency (Hz)")
    ylabel = f"SPL ({'dBA' if weighted else 'dBZ'})"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=PLOT_GRID_LINEWIDTH)
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(y_min, y_max)

    # Use fixed locator for log scale if default is problematic
    # ax.xaxis.set_major_locator(mticker.LogLocator(numticks=15)) # Example
    # ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=15, subs=np.arange(2, 10) * .1)) # Example

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) # Use ScalarFormatter for non-scientific notation
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    # Ensure ticks are generated before setting labels
    fig.canvas.draw() # Force drawing to generate ticks
    plt.setp(ax.get_xticklabels(), rotation=PLOT_XTICK_ROTATION, ha='right')


    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.close(fig)


def plot_third_octave_spl(center_freqs, octave_spl, title, output_filename, width, height, weighted, color=None,
                          freq_min=PLOT_FREQ_MIN_HZ, freq_max=PLOT_FREQ_MAX_HZ,
                          y_min=PLOT_OCTAVE_Y_MIN_DB, y_max=PLOT_OCTAVE_Y_MAX_DB):
    """
    Creates and saves a 1/3 octave SPL bar chart.

    Args:
        center_freqs (np.array): Center frequencies of the 1/3 octave bands.
        octave_spl (np.array): SPL data (dBA or dBZ) for each band (can contain -inf).
        title (str): Plot title.
        output_filename (str): Full path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        weighted (bool): If True, label y-axis as dBA, else dBZ.
        color (str, optional): Specific color for the bars. Defaults to 'skyblue'.
        freq_min (float, optional): Min frequency to consider for band inclusion. Defaults to PLOT_FREQ_MIN_HZ.
        freq_max (float, optional): Max frequency to consider for band inclusion. Defaults to PLOT_FREQ_MAX_HZ.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_OCTAVE_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_OCTAVE_Y_MAX_DB.
    """
    fig, ax = plt.subplots(figsize=(width, height))

    # --- Filtering Logic ---
    all_indices = np.arange(len(center_freqs))
    factor = 2**(1/6)
    lower_limits_all = center_freqs / factor
    upper_limits_all = center_freqs * factor
    # Include bands that *overlap* the frequency range
    valid_indices_freq_range = all_indices[
        (upper_limits_all > freq_min) & (lower_limits_all < freq_max)
    ]

    if len(valid_indices_freq_range) == 0:
         print(f"Warning: No 1/3 octave bands overlap the range {freq_min} Hz to {freq_max} Hz.")
         plt.close(fig)
         return

    # Filter SPL data based on frequency range first
    spl_in_range = octave_spl[valid_indices_freq_range]
    # Now find indices within this subset that have finite SPL
    valid_indices_spl_rel = np.where(np.isfinite(spl_in_range))[0]

    if len(valid_indices_spl_rel) == 0:
        print(f"Warning: No finite SPL data in 1/3 octave bands within the range {freq_min}-{freq_max} Hz to plot for {title}")
        plt.close(fig)
        return

    # Get the absolute indices corresponding to valid frequency AND finite SPL
    final_indices = valid_indices_freq_range[valid_indices_spl_rel]
    plot_center_freqs = center_freqs[final_indices]
    plot_levels = octave_spl[final_indices]
    # --- End Filtering Logic ---

    num_freqs = len(plot_center_freqs)
    x_indices = np.arange(num_freqs)
    bar_width = 0.8

    ax.set_ylim(y_min, y_max)
    # Clip finite values to plot range
    plot_levels_clipped = np.clip(plot_levels, y_min, y_max)
    plot_color = color if color else 'skyblue'

    ax.bar(
        x_indices,
        plot_levels_clipped - y_min, # Height of bar above y_min
        bar_width,
        bottom=y_min, # Start bar at y_min
        color=plot_color,
        edgecolor='black',
        linewidth=PLOT_BAR_LINEWIDTH,
        zorder=2
    )

    ax.set_xlabel("Frequency (Hz)")
    ylabel = f"1/3 Octave Band SPL ({'dBA' if weighted else 'dBZ'})"
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{f:g}" for f in plot_center_freqs], rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.tick_params(axis='x', which='minor', bottom=False)

    if num_freqs > 0:
        ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
    else:
        ax.set_xlim(-0.5, 0.5) # Handle case with no bars

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.close(fig)


def plot_combined_narrowband_spl(all_data, title, output_filename, width, height, weighted,
                                 freq_min=PLOT_FREQ_MIN_HZ, freq_max=PLOT_FREQ_MAX_HZ,
                                 y_min=PLOT_NB_Y_MIN_DB, y_max=PLOT_NB_Y_MAX_DB):
    """
    Creates and saves a combined narrow-band SPL spectrum plot for multiple datasets (e.g., settings).

    Args:
        all_data (list): A list of tuples, where each tuple is (label, frequencies, spl_levels).
                         spl_levels are dBA or dBZ based on 'weighted' flag.
        title (str): The main title for the plot.
        output_filename (str): Path to save the SVG file.
        width (float): Width of the plot in inches.
        height (float): Height of the plot in inches.
        weighted (bool): Whether the SPL data is A-weighted (for labeling).
        freq_min (float, optional): Minimum frequency for plot x-axis. Defaults to PLOT_FREQ_MIN_HZ.
        freq_max (float, optional): Maximum frequency for plot x-axis. Defaults to PLOT_FREQ_MAX_HZ.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_NB_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_NB_Y_MAX_DB.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xscale('log')
    ax.set_xlabel("Frequency (Hz)")
    ylabel = f"SPL ({'dBA' if weighted else 'dBZ'})"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=PLOT_GRID_LINEWIDTH)
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(y_min, y_max)

    plotted_something = False

    try:
        # Use the helper function from this library for sorting
        all_data_sorted = sorted(all_data, key=lambda item: get_setting_sort_key(item[0]))
    except Exception as e:
        print(f"Warning: Could not sort labels reliably ({e}). Using original order.")
        all_data_sorted = all_data

    # Let Matplotlib cycle through colors
    for label, frequencies, spl_levels in all_data_sorted:
        # Plot only finite SPL values within the frequency range
        plot_indices = np.where((frequencies >= freq_min) & (frequencies <= freq_max) & np.isfinite(spl_levels))[0]

        if len(plot_indices) > 0:
            plot_freqs = frequencies[plot_indices]
            plot_spl = spl_levels[plot_indices]
            ax.plot(plot_freqs, plot_spl, linewidth=PLOT_LINEWIDTH, label=f"Setting {label}")
            plotted_something = True
        else:
            print(f"Warning: No finite data points between {freq_min}-{freq_max} Hz for Label '{label}' in combined plot.")

    if not plotted_something:
        print(f"Warning: No valid data found across all datasets to plot for {title}")
        plt.close(fig)
        return

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    # Ensure ticks are generated before setting labels
    fig.canvas.draw()
    plt.setp(ax.get_xticklabels(), rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.legend(fontsize='small')

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Combined narrow-band plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving combined plot {output_filename}: {e}")
    plt.close(fig)


def plot_comparison_narrowband_spl(
    data_by_label, # Dict: {label: (frequencies, spl_levels)}
    title,
    output_filename,
    width,
    height,
    weighted,
    label_color_map, # Dict: {label: color_string}
    freq_min=PLOT_FREQ_MIN_HZ, freq_max=PLOT_FREQ_MAX_HZ,
    y_min=PLOT_NB_Y_MIN_DB, y_max=PLOT_NB_Y_MAX_DB
):
    """
    Creates and saves a comparison narrow-band SPL spectrum plot for multiple datasets.

    Args:
        data_by_label (dict): {label: (frequencies, spl_levels)}.
                               spl_levels are dBA or dBZ based on 'weighted' flag.
        title (str): The main title for the plot.
        output_filename (str): Path to save the SVG file.
        width (float): Width of the plot in inches.
        height (float): Height of the plot in inches.
        weighted (bool): Whether the SPL data is A-weighted (for labeling).
        label_color_map (dict): {label: color_string}. Maps labels to plot colors.
        freq_min (float, optional): Minimum frequency for plot x-axis. Defaults to PLOT_FREQ_MIN_HZ.
        freq_max (float, optional): Maximum frequency for plot x-axis. Defaults to PLOT_FREQ_MAX_HZ.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_NB_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_NB_Y_MAX_DB.
    """
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xscale('log')
    ax.set_xlabel("Frequency (Hz)")
    ylabel = f"SPL ({'dBA' if weighted else 'dBZ'})"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=PLOT_GRID_LINEWIDTH)
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(y_min, y_max)

    plotted_something = False
    labels = sorted(data_by_label.keys()) # Consistent legend order

    for label in labels:
        frequencies, spl_levels = data_by_label[label]
        # Plot only finite SPL values within the frequency range
        plot_indices = np.where(
            (frequencies >= freq_min) & (frequencies <= freq_max) & np.isfinite(spl_levels)
        )[0]

        if len(plot_indices) > 0:
            plot_freqs = frequencies[plot_indices]
            plot_spl = spl_levels[plot_indices]
            label_color = label_color_map.get(label, None) # Use None for default if not found
            ax.plot(plot_freqs, plot_spl, linewidth=PLOT_LINEWIDTH, label=label, color=label_color)
            plotted_something = True
        else:
            print(f"Warning: No finite data points between {freq_min}-{freq_max} Hz for Label '{label}' in comparison plot.")

    if not plotted_something:
        print(f"Warning: No valid data found across all labels to plot for '{title}'. Skipping save.")
        plt.close(fig)
        return

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    # Ensure ticks are generated before setting labels
    fig.canvas.draw()
    plt.setp(ax.get_xticklabels(), rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.legend(fontsize='small')

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Comparison narrow-band plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving comparison narrow-band plot {output_filename}: {e}")
    plt.close(fig)


def plot_grouped_third_octave_spl(
    data_by_label, # Dict: {label: octave_spl_array}
    nominal_frequencies,
    title,
    output_filename,
    width,
    height,
    weighted,
    label_color_map, # Dict: {label: color_string}
    freq_min=PLOT_FREQ_MIN_HZ, freq_max=PLOT_FREQ_MAX_HZ,
    y_min=PLOT_OCTAVE_Y_MIN_DB, y_max=PLOT_OCTAVE_Y_MAX_DB
):
    """
    Creates a grouped 1/3 octave bar chart comparing multiple datasets.

    Args:
        data_by_label (dict): {label: octave_spl_array}. Labels map to octave SPL arrays (dBA or dBZ).
        nominal_frequencies (np.array): The center frequencies for the bands.
        title (str): Plot title.
        output_filename (str): Path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        weighted (bool): Whether data is A-weighted (for label).
        label_color_map (dict): {label: color_string}. Maps labels to bar colors.
        freq_min (float, optional): Min frequency to consider for band inclusion. Defaults to PLOT_FREQ_MIN_HZ.
        freq_max (float, optional): Max frequency to consider for band inclusion. Defaults to PLOT_FREQ_MAX_HZ.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_OCTAVE_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_OCTAVE_Y_MAX_DB.
    """
    labels = sorted(data_by_label.keys()) # Consistent order
    num_labels = len(labels)
    if num_labels == 0:
        print(f"Warning: No data provided for grouped plot '{title}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(width, height))

    # --- Filter frequencies and corresponding data ---
    factor = 2**(1/6)
    lower_limits_all = nominal_frequencies / factor
    upper_limits_all = nominal_frequencies * factor
    # Include bands that *overlap* the frequency range
    plot_band_indices = np.where(
        (upper_limits_all > freq_min) & (lower_limits_all < freq_max)
    )[0]

    if len(plot_band_indices) == 0:
        print(f"Warning: No 1/3 octave bands overlap the plot range {freq_min}-{freq_max} Hz for '{title}'. Skipping.")
        plt.close(fig)
        return

    plot_freqs = nominal_frequencies[plot_band_indices]
    num_freqs = len(plot_freqs)
    x_indices = np.arange(num_freqs)

    total_group_width = 0.8
    bar_width = total_group_width / num_labels
    group_start_offset = -total_group_width / 2

    ax.set_ylim(y_min, y_max)
    plotted_something = False

    for i, label in enumerate(labels):
        octave_spl = data_by_label[label]
        # Ensure octave_spl has data for all nominal frequencies before indexing
        if len(octave_spl) != len(nominal_frequencies):
             print(f"Warning: Mismatch in octave SPL length for label '{label}' in grouped plot. Skipping.")
             continue # Skip this label if data length is wrong

        # Filter data for the bands within the plot's frequency range
        plot_spl_data_full = octave_spl[plot_band_indices]
        # Replace non-finite with NaN before clipping
        plot_spl_data_proc = np.where(np.isfinite(plot_spl_data_full), plot_spl_data_full, np.nan)
        # Clip finite values to plot range
        plot_spl_data_clipped = np.clip(plot_spl_data_proc, y_min, y_max)
        bar_positions = x_indices + group_start_offset + (i * bar_width) + bar_width / 2
        label_color = label_color_map.get(label, None)

        # Only plot if there's at least one non-NaN value after processing
        if not np.all(np.isnan(plot_spl_data_clipped)):
            # Calculate bar heights relative to y_min
            bar_heights = np.nan_to_num(plot_spl_data_clipped, nan=y_min) - y_min
            bar_heights[bar_heights < 0] = 0 # Ensure height is not negative

            ax.bar(
                bar_positions,
                bar_heights, # Height above y_min
                bar_width,
                bottom=y_min, # Start bar at y_min
                label=label,
                color=label_color,
                linewidth=PLOT_BAR_LINEWIDTH,
                edgecolor='black',
                zorder=2
            )
            plotted_something = True

    if not plotted_something:
        print(f"Warning: No valid SPL data found within plot range/limits for grouped plot '{title}'. Skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Frequency (Hz)")
    ylabel = f"1/3 Octave Band SPL ({'dBA' if weighted else 'dBZ'})"
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{f:g}" for f in plot_freqs], rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.tick_params(axis='x', which='minor', bottom=False)
    if num_freqs > 0:
        ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
    else:
        ax.set_xlim(-0.5, 0.5)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)
    ax.legend(fontsize='small')

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Grouped comparison plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving grouped plot {output_filename}: {e}")
    plt.close(fig)

def plot_articulation_index_by_setting(
    vehicle_name,
    ai_data, # Dict: {setting_label: ai_value}
    output_filename,
    width,
    height,
    color=None
):
    """
    Creates a bar plot showing Articulation Index vs. Blower Setting for a vehicle.

    Args:
        vehicle_name (str): Name of the vehicle for the title.
        ai_data (dict): Dictionary mapping setting labels to AI values.
        output_filename (str): Full path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        color (str, optional): Specific color for the bars. Defaults to 'skyblue'.
    """
    if not ai_data:
        print(f"Warning: No AI data provided for vehicle '{vehicle_name}'. Skipping plot.")
        return

    # Filter out any potential NaN/None values before sorting/plotting
    valid_ai_data = {k: v for k, v in ai_data.items() if v is not None and np.isfinite(v)}

    if not valid_ai_data:
        print(f"Warning: No valid (finite) AI data provided for vehicle '{vehicle_name}'. Skipping plot.")
        return

    # Sort settings using the library's sort key function
    try:
        sorted_settings = sorted(valid_ai_data.keys(), key=get_setting_sort_key)
    except Exception as e:
        print(f"Warning: Could not sort settings for AI plot of '{vehicle_name}' ({e}). Using default order.")
        sorted_settings = list(valid_ai_data.keys())

    ai_values = [valid_ai_data[setting] for setting in sorted_settings]
    setting_labels = [str(s) for s in sorted_settings] # Ensure labels are strings

    fig, ax = plt.subplots(figsize=(width, height))
    x_indices = np.arange(len(setting_labels))
    bar_width = 0.6
    plot_color = color if color else 'skyblue'

    ax.bar(
        x_indices,
        ai_values,
        bar_width,
        color=plot_color,
        edgecolor='black',
        linewidth=PLOT_BAR_LINEWIDTH,
        zorder=2
    )

    ax.set_xlabel("Blower Setting")
    ax.set_ylabel("Articulation Index (AI) (%)")
    ax.set_title(f"{vehicle_name} - Articulation Index (AI) vs. Blower Setting")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(setting_labels)
    ax.set_ylim(0, 105) # AI ranges from 0 to 100 (allow slight overshoot)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False) # No vertical grid lines usually needed for category bars

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"AI plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving AI plot {output_filename}: {e}")
    plt.close(fig)


def plot_ai_band_details(
    ai_details, # Dictionary from get_ai_calculation_details
    title,
    output_filename,
    width,
    height,
    color=None,
    y_min=PLOT_AI_DETAIL_Y_MIN_DB,
    y_max=PLOT_AI_DETAIL_Y_MAX_DB
):
    """
    Creates a bar plot showing A-weighted SPL and AI limits for AI frequency bands.

    Args:
        ai_details (dict): Dictionary returned by get_ai_calculation_details.
        title (str): Plot title.
        output_filename (str): Full path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        color (str, optional): Specific color for the bars. Defaults to 'skyblue'.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_AI_DETAIL_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_AI_DETAIL_Y_MAX_DB.
    """
    # Extract data, ensuring consistent order based on AI_FREQUENCIES
    plot_freqs = AI_FREQUENCIES
    spl_values = np.array([ai_details['spl_values'].get(f, np.nan) for f in plot_freqs])
    lower_limits = np.array([ai_details['lower_limits'].get(f, np.nan) for f in plot_freqs])
    upper_limits = np.array([ai_details['upper_limits'].get(f, np.nan) for f in plot_freqs])
    total_ai = ai_details['total_ai']

    # Filter out any bands where we have no data at all (e.g., if warning occurred in calc)
    valid_band_indices = np.where(np.isfinite(spl_values) & np.isfinite(lower_limits) & np.isfinite(upper_limits))[0]

    if len(valid_band_indices) == 0:
        print(f"Warning: No valid AI band data to plot for '{title}'. Skipping plot.")
        return

    plot_freqs = plot_freqs[valid_band_indices]
    spl_values = spl_values[valid_band_indices]
    lower_limits = lower_limits[valid_band_indices]
    upper_limits = upper_limits[valid_band_indices]

    num_freqs = len(plot_freqs)
    x_indices = np.arange(num_freqs)
    bar_width = 0.8
    limit_line_width = bar_width * 0.8 # Make limit lines slightly narrower than bars
    limit_line_offset = limit_line_width / 2

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_ylim(y_min, y_max)

    # Clip finite values to plot range
    spl_values_clipped = np.clip(spl_values, y_min, y_max)
    plot_color = color if color else 'skyblue'

    # Plot SPL bars
    # Calculate bar heights relative to y_min
    bar_heights = spl_values_clipped - y_min
    bar_heights[bar_heights < 0] = 0 # Ensure height is not negative

    bars = ax.bar(
        x_indices,
        bar_heights, # Height above y_min
        bar_width,
        bottom=y_min, # Start bar at y_min
        color=plot_color,
        edgecolor='black',
        linewidth=PLOT_BAR_LINEWIDTH,
        zorder=2,
        label='Band SPL (dBA)' # Add label for legend
    )

    # Plot AI Limit Lines using hlines for clarity
    limit_lines = []
    # Keep track of handles added to avoid duplicates in legend
    added_ll_legend = False
    added_ul_legend = False
    for i in range(num_freqs):
        x_pos = x_indices[i]
        # Plot lower limit line segment
        if np.isfinite(lower_limits[i]):
            ll = ax.hlines(lower_limits[i], x_pos - limit_line_offset, x_pos + limit_line_offset,
                       colors='red', linestyles='solid', linewidth=PLOT_LINEWIDTH * 1.5, zorder=3)
            if not added_ll_legend:
                 limit_lines.append(ll) # Add only once for legend
                 added_ll_legend = True
        # Plot upper limit line segment
        if np.isfinite(upper_limits[i]):
            ul = ax.hlines(upper_limits[i], x_pos - limit_line_offset, x_pos + limit_line_offset,
                       colors='green', linestyles='solid', linewidth=PLOT_LINEWIDTH * 1.5, zorder=3)
            if not added_ul_legend:
                 limit_lines.append(ul) # Add only once for legend
                 added_ul_legend = True

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("1/3 Octave Band SPL (dBA)")
    # Include total AI in the title
    full_title = f"{title}\nTotal AI: {total_ai:.1f}%"
    ax.set_title(full_title)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{f:g}" for f in plot_freqs], rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.tick_params(axis='x', which='minor', bottom=False)

    if num_freqs > 0:
        ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
    else:
        ax.set_xlim(-0.5, 0.5)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)

    # Add legend for limit lines
    if limit_lines:
        # Get the bar artist for the legend too
        legend_handles = [bars] + limit_lines
        legend_labels = ['Band SPL (dBA)']
        if added_ll_legend: legend_labels.append('AI Lower Limit')
        if added_ul_legend: legend_labels.append('AI Upper Limit')
        ax.legend(handles=legend_handles, labels=legend_labels, fontsize='small')


    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"AI Detail plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving AI Detail plot {output_filename}: {e}")
    plt.close(fig)

# --- NEW AI DETAIL COMPARISON PLOTTING FUNCTION ---
def plot_comparison_ai_band_details(
    data_by_vehicle, # Dict: {vehicle_name: ai_details_dict} for ONE setting
    title,
    output_filename,
    width,
    height,
    vehicle_color_map, # Dict: {vehicle_name: color_string}
    y_min=PLOT_AI_DETAIL_Y_MIN_DB, # Reuse existing AI detail limits
    y_max=PLOT_AI_DETAIL_Y_MAX_DB
):
    """
    Creates a grouped bar chart comparing AI Band Details (SPL, Limits)
    across multiple vehicles for a single setting.

    Args:
        data_by_vehicle (dict): {vehicle_name: ai_details_dict}.
                                ai_details_dict is the output from
                                get_ai_calculation_details for one setting.
        title (str): Plot title.
        output_filename (str): Path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        vehicle_color_map (dict): {vehicle_name: color_string}. Maps vehicles to bar colors.
        y_min (float, optional): Minimum SPL for plot y-axis. Defaults to PLOT_AI_DETAIL_Y_MIN_DB.
        y_max (float, optional): Maximum SPL for plot y-axis. Defaults to PLOT_AI_DETAIL_Y_MAX_DB.
    """
    vehicle_names = sorted(data_by_vehicle.keys()) # Consistent order
    num_vehicles = len(vehicle_names)
    if num_vehicles == 0:
        print(f"Warning: No AI detail data provided for comparison plot '{title}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(width, height))

    # --- Determine Frequency Bands and Filter ---
    # Use the standard AI frequencies as the basis
    plot_freqs = AI_FREQUENCIES
    num_freqs = len(plot_freqs)
    if num_freqs == 0:
        print(f"Warning: No AI frequencies defined. Cannot create AI detail comparison plot '{title}'.")
        plt.close(fig)
        return

    x_indices = np.arange(num_freqs) # One group per AI frequency band

    # --- Define Bar/Group Widths EARLY --- <<< MOVED UP
    total_group_width = 0.8
    bar_width = total_group_width / num_vehicles
    group_start_offset = -total_group_width / 2

    # --- Plot AI Limit Lines (Once per band) ---
    # Extract limits from the standard AI parameters
    lower_limits_std = np.array([AI_PARAMETERS.get(f, (np.nan,))[0] for f in plot_freqs])
    upper_limits_std = np.array([AI_PARAMETERS.get(f, (np.nan, np.nan))[1] for f in plot_freqs])

    limit_line_width_factor = 0.9
    limit_line_offset = (total_group_width * limit_line_width_factor) / 2

    limit_lines_handles = [] # Handles specifically for legend
    added_ll_legend = False
    added_ul_legend = False
    for i in range(num_freqs):
        x_pos = x_indices[i]
        # Plot lower limit line segment across the group width
        if np.isfinite(lower_limits_std[i]):
            # *** FIX: Add label here, use "_nolegend_" after the first one ***
            ll_label = 'AI Lower Limit' if not added_ll_legend else "_nolegend_"
            ll = ax.hlines(lower_limits_std[i], x_pos - limit_line_offset, x_pos + limit_line_offset,
                       colors='red', linestyles='solid', linewidth=PLOT_LINEWIDTH * 1.5, zorder=3,
                       label=ll_label) # <-- ADDED LABEL ARGUMENT
            if not added_ll_legend:
                 limit_lines_handles.append(ll) # Add handle only once for legend
                 added_ll_legend = True
        # Plot upper limit line segment across the group width
        if np.isfinite(upper_limits_std[i]):
            # *** FIX: Add label here, use "_nolegend_" after the first one ***
            ul_label = 'AI Upper Limit' if not added_ul_legend else "_nolegend_"
            ul = ax.hlines(upper_limits_std[i], x_pos - limit_line_offset, x_pos + limit_line_offset,
                       colors='green', linestyles='solid', linewidth=PLOT_LINEWIDTH * 1.5, zorder=3,
                       label=ul_label) # <-- ADDED LABEL ARGUMENT
            if not added_ul_legend:
                 limit_lines_handles.append(ul) # Add handle only once for legend
                 added_ul_legend = True


    # --- Plot Grouped Bars for Vehicle SPLs ---
    ax.set_ylim(y_min, y_max)
    plotted_something = False
    bar_handles = {} # FIXED: Use a dictionary to store handles by vehicle name

    for i, vehicle_name in enumerate(vehicle_names):
        ai_details = data_by_vehicle[vehicle_name]
        # Extract SPL values for this vehicle, ensuring order matches plot_freqs
        spl_values_vehicle = np.array([ai_details['spl_values'].get(f, np.nan) for f in plot_freqs])

        # Process NaN/Inf before clipping/plotting
        plot_data_proc = np.where(np.isfinite(spl_values_vehicle), spl_values_vehicle, np.nan)
        # Clip finite values to plot range
        plot_data_clipped = np.clip(plot_data_proc, y_min, y_max)

        bar_positions = x_indices + group_start_offset + (i * bar_width) + bar_width / 2
        vehicle_color = vehicle_color_map.get(vehicle_name, None) # Get color, default to None

        # Only plot if there's at least one non-NaN value after processing
        if not np.all(np.isnan(plot_data_clipped)):
            # Calculate bar heights relative to y_min
            bar_heights = np.nan_to_num(plot_data_clipped, nan=y_min) - y_min
            bar_heights[bar_heights < 0] = 0 # Ensure height is not negative

            bars = ax.bar(
                bar_positions,
                bar_heights,
                bar_width,
                bottom=y_min,
                label=vehicle_name, # Correctly set here
                color=vehicle_color,
                linewidth=PLOT_BAR_LINEWIDTH,
                edgecolor='black',
                zorder=2
            )
            # FIXED: Store the handle with the vehicle name as key
            bar_handles[vehicle_name] = bars[0]
            plotted_something = True

    if not plotted_something:
        print(f"Warning: No valid SPL data found within plot range/limits for AI detail comparison plot '{title}'. Skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("1/3 Octave Band SPL (dBA)") # AI Details always use dBA
    ax.set_title(title)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{f:g}" for f in plot_freqs], rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.tick_params(axis='x', which='minor', bottom=False)
    if num_freqs > 0:
        ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
    else:
        ax.set_xlim(-0.5, 0.5)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)

    # --- Create Legend ---
    # FIXED: Create a list of handles in the same order as vehicle_names
    all_legend_handles = [bar_handles[name] for name in vehicle_names if name in bar_handles]
    all_legend_handles.extend(limit_lines_handles)
    
    # Check if we actually have handles to avoid errors/warnings
    if all_legend_handles:
        # FIXED: Let matplotlib use the labels from the artists
        ax.legend(fontsize='small')

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"AI Detail comparison plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving AI Detail comparison plot {output_filename}: {e}")
    plt.close(fig)

# --- NEW LOUDNESS PLOTTING FUNCTION ---
def plot_loudness_by_setting(
    vehicle_name,
    loudness_data, # Dict: {setting_label: loudness_value_sones}
    output_filename,
    width,
    height,
    color=None,
    y_min=PLOT_LOUDNESS_Y_MIN # Use constant for min y
):
    """
    Creates a bar plot showing Loudness (Sones) vs. Blower Setting for a vehicle.

    Args:
        vehicle_name (str): Name of the vehicle for the title.
        loudness_data (dict): Dictionary mapping setting labels to Loudness values (Sones).
        output_filename (str): Full path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        color (str, optional): Specific color for the bars. Defaults to 'skyblue'.
        y_min (float, optional): Minimum Loudness for plot y-axis. Defaults to PLOT_LOUDNESS_Y_MIN.
    """
    if not loudness_data:
        print(f"Warning: No Loudness data provided for vehicle '{vehicle_name}'. Skipping plot.")
        return

    # Filter out any potential NaN/None/non-finite values before sorting/plotting
    valid_loudness_data = {k: v for k, v in loudness_data.items() if v is not None and np.isfinite(v)}

    if not valid_loudness_data:
        print(f"Warning: No valid (finite) Loudness data provided for vehicle '{vehicle_name}'. Skipping plot.")
        return

    # Sort settings using the library's sort key function
    try:
        sorted_settings = sorted(valid_loudness_data.keys(), key=get_setting_sort_key)
    except Exception as e:
        print(f"Warning: Could not sort settings for Loudness plot of '{vehicle_name}' ({e}). Using default order.")
        sorted_settings = list(valid_loudness_data.keys())

    loudness_values = [valid_loudness_data[setting] for setting in sorted_settings]
    setting_labels = [str(s) for s in sorted_settings] # Ensure labels are strings

    fig, ax = plt.subplots(figsize=(width, height))
    x_indices = np.arange(len(setting_labels))
    bar_width = 0.6
    plot_color = color if color else 'skyblue'

    ax.bar(
        x_indices,
        loudness_values,
        bar_width,
        color=plot_color,
        edgecolor='black',
        linewidth=PLOT_BAR_LINEWIDTH,
        zorder=2
    )

    ax.set_xlabel("Blower Setting")
    ax.set_ylabel("Loudness (Sones)")
    ax.set_title(f"{vehicle_name} - Loudness vs. Blower Setting")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(setting_labels)
    ax.set_ylim(bottom=y_min) # Start y-axis at defined minimum (usually 0)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Loudness plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving Loudness plot {output_filename}: {e}")
    plt.close(fig)


def plot_comparison_loudness_by_setting(
    all_vehicles_loudness_data, # Dict: {vehicle_name: {setting: sone_value}}
    title,
    output_filename,
    width,
    height,
    vehicle_color_map, # Dict: {vehicle_name: color_string}
    y_min=PLOT_LOUDNESS_Y_MIN # Use existing constant for y-min
    # y_max can be determined automatically or set via a new constant if needed
):
    """
    Creates a grouped bar chart comparing Loudness (Sones) across multiple
    vehicles for each common setting.

    Args:
        all_vehicles_loudness_data (dict): {vehicle: {setting: sone_value}}.
        title (str): Plot title.
        output_filename (str): Path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        vehicle_color_map (dict): {vehicle_name: color_string}. Maps vehicles to bar colors.
        y_min (float, optional): Minimum Loudness for plot y-axis. Defaults to PLOT_LOUDNESS_Y_MIN.
    """
    vehicle_names = sorted(all_vehicles_loudness_data.keys()) # Consistent order
    num_vehicles = len(vehicle_names)
    if num_vehicles == 0:
        print(f"Warning: No loudness data provided for comparison plot '{title}'. Skipping.")
        return

    # Find all unique settings across all vehicles with valid data
    all_settings = set()
    for vehicle_name in vehicle_names:
        vehicle_data = all_vehicles_loudness_data.get(vehicle_name, {})
        if vehicle_data: # Check if data exists for the vehicle
             # Add settings only if the corresponding value is finite
             all_settings.update({s for s, v in vehicle_data.items() if v is not None and np.isfinite(v)})

    if not all_settings:
        print(f"Warning: No valid settings with finite loudness data found across vehicles for '{title}'. Skipping.")
        return

    # Sort settings using the library's sort key function
    try:
        sorted_settings = sorted(list(all_settings), key=get_setting_sort_key)
    except Exception as e:
        print(f"Warning: Could not sort settings for Loudness comparison plot '{title}' ({e}). Using default order.")
        sorted_settings = sorted(list(all_settings))

    num_settings = len(sorted_settings)
    if num_settings == 0:
        print(f"Warning: No settings left after sorting/filtering for '{title}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(width, height))

    x_indices = np.arange(num_settings) # One group per setting
    total_group_width = 0.8
    bar_width = total_group_width / num_vehicles
    group_start_offset = -total_group_width / 2

    # Determine y_max dynamically based on valid data, add some padding
    max_loudness = y_min # Start with y_min
    found_valid_data = False
    for setting in sorted_settings:
        for vehicle_name in vehicle_names:
            loudness_val = all_vehicles_loudness_data.get(vehicle_name, {}).get(setting, np.nan)
            if np.isfinite(loudness_val):
                max_loudness = max(max_loudness, loudness_val)
                found_valid_data = True

    if not found_valid_data:
        print(f"Warning: No finite loudness values found across all vehicles and settings for '{title}'. Skipping plot.")
        plt.close(fig)
        return

    y_max_plot = max_loudness * 1.05 if max_loudness > y_min else y_min + 1 # Add 5% padding, ensure range > 0
    ax.set_ylim(y_min, y_max_plot)

    plotted_something = False
    for i, vehicle_name in enumerate(vehicle_names):
        loudness_values_for_vehicle = []
        for setting in sorted_settings:
            # Get loudness, default to NaN if vehicle or setting is missing
            value = all_vehicles_loudness_data.get(vehicle_name, {}).get(setting, np.nan)
            loudness_values_for_vehicle.append(value)

        loudness_values_np = np.array(loudness_values_for_vehicle, dtype=float)

        # Process NaN/Inf before clipping/plotting
        plot_data_proc = np.where(np.isfinite(loudness_values_np), loudness_values_np, np.nan)
        # Clip finite values to plot range (using calculated y_max_plot)
        plot_data_clipped = np.clip(plot_data_proc, y_min, y_max_plot)

        bar_positions = x_indices + group_start_offset + (i * bar_width) + bar_width / 2
        vehicle_color = vehicle_color_map.get(vehicle_name, None) # Get color, default to None

        # Only plot if there's at least one non-NaN value after processing
        if not np.all(np.isnan(plot_data_clipped)):
            # Calculate bar heights relative to y_min
            # Use nan_to_num to handle potential NaNs remaining after clipping (though unlikely)
            bar_heights = np.nan_to_num(plot_data_clipped, nan=y_min) - y_min
            bar_heights[bar_heights < 0] = 0 # Ensure height is not negative

            ax.bar(
                bar_positions,
                bar_heights, # Height above y_min
                bar_width,
                bottom=y_min, # Start bar at y_min
                label=vehicle_name,
                color=vehicle_color,
                linewidth=PLOT_BAR_LINEWIDTH,
                edgecolor='black',
                zorder=2
            )
            plotted_something = True

    if not plotted_something:
        print(f"Warning: No valid Loudness data found within plot range/limits for comparison plot '{title}'. Skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Blower Setting")
    ax.set_ylabel("Loudness (Sones)")
    ax.set_title(title)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in sorted_settings]) # Use sorted setting labels
    ax.tick_params(axis='x', which='minor', bottom=False)
    if num_settings > 0:
        ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
    else:
        ax.set_xlim(-0.5, 0.5)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)
    ax.legend(fontsize='small')

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"Loudness comparison plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving Loudness comparison plot {output_filename}: {e}")
    plt.close(fig)



def plot_comparison_ai_by_setting(
    all_vehicles_ai_data, # Dict: {vehicle_name: {setting: ai_value}}
    title,
    output_filename,
    width,
    height,
    vehicle_color_map, # Dict: {vehicle_name: color_string}
    y_min=PLOT_AI_COMP_Y_MIN, # Use new constant for min y
    y_max=PLOT_AI_COMP_Y_MAX  # Use new constant for max y
):
    """
    Creates a grouped bar chart comparing Articulation Index (AI) across multiple
    vehicles for each common setting.

    Args:
        all_vehicles_ai_data (dict): {vehicle: {setting: ai_value}}.
        title (str): Plot title.
        output_filename (str): Path to save the SVG file.
        width (float): Plot width in inches.
        height (float): Plot height in inches.
        vehicle_color_map (dict): {vehicle_name: color_string}. Maps vehicles to bar colors.
        y_min (float, optional): Minimum AI for plot y-axis. Defaults to PLOT_AI_COMP_Y_MIN.
        y_max (float, optional): Maximum AI for plot y-axis. Defaults to PLOT_AI_COMP_Y_MAX.
    """
    vehicle_names = sorted(all_vehicles_ai_data.keys()) # Consistent order
    num_vehicles = len(vehicle_names)
    if num_vehicles == 0:
        print(f"Warning: No AI data provided for comparison plot '{title}'. Skipping.")
        return

    # Find all unique settings across all vehicles with valid (finite) AI data
    all_settings = set()
    found_valid_data_overall = False
    for vehicle_name in vehicle_names:
        vehicle_data = all_vehicles_ai_data.get(vehicle_name, {})
        if vehicle_data: # Check if data exists for the vehicle
             # Add settings only if the corresponding value is finite
             valid_settings_for_vehicle = {s for s, v in vehicle_data.items() if v is not None and np.isfinite(v)}
             if valid_settings_for_vehicle:
                 all_settings.update(valid_settings_for_vehicle)
                 found_valid_data_overall = True # Mark that we found at least one valid point

    if not found_valid_data_overall:
        print(f"Warning: No valid (finite) AI data found across any vehicles for '{title}'. Skipping plot.")
        return
    if not all_settings:
        print(f"Warning: No settings with finite AI data found across vehicles for '{title}'. Skipping.")
        return

    # Sort settings using the library's sort key function
    try:
        sorted_settings = sorted(list(all_settings), key=get_setting_sort_key)
    except Exception as e:
        print(f"Warning: Could not sort settings for AI comparison plot '{title}' ({e}). Using default order.")
        sorted_settings = sorted(list(all_settings))

    num_settings = len(sorted_settings)
    if num_settings == 0:
        print(f"Warning: No settings left after sorting/filtering for '{title}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(width, height))

    x_indices = np.arange(num_settings) # One group per setting
    total_group_width = 0.8
    bar_width = total_group_width / num_vehicles
    group_start_offset = -total_group_width / 2

    # Use the provided y_min and y_max for AI
    ax.set_ylim(y_min, y_max)

    plotted_something = False
    for i, vehicle_name in enumerate(vehicle_names):
        ai_values_for_vehicle = []
        for setting in sorted_settings:
            # Get AI value, default to NaN if vehicle or setting is missing or invalid
            value = all_vehicles_ai_data.get(vehicle_name, {}).get(setting, np.nan)
            ai_values_for_vehicle.append(value)

        ai_values_np = np.array(ai_values_for_vehicle, dtype=float)

        # Process NaN/Inf before clipping/plotting
        plot_data_proc = np.where(np.isfinite(ai_values_np), ai_values_np, np.nan)
        # Clip finite values to plot range
        plot_data_clipped = np.clip(plot_data_proc, y_min, y_max)

        bar_positions = x_indices + group_start_offset + (i * bar_width) + bar_width / 2
        vehicle_color = vehicle_color_map.get(vehicle_name, None) # Get color, default to None

        # Only plot if there's at least one non-NaN value after processing
        if not np.all(np.isnan(plot_data_clipped)):
            # Calculate bar heights relative to y_min
            # Use nan_to_num to handle potential NaNs remaining after clipping
            bar_heights = np.nan_to_num(plot_data_clipped, nan=y_min) - y_min
            bar_heights[bar_heights < 0] = 0 # Ensure height is not negative

            ax.bar(
                bar_positions,
                bar_heights, # Height above y_min
                bar_width,
                bottom=y_min, # Start bar at y_min
                label=vehicle_name,
                color=vehicle_color,
                linewidth=PLOT_BAR_LINEWIDTH,
                edgecolor='black',
                zorder=2
            )
            plotted_something = True

    if not plotted_something:
        print(f"Warning: No valid AI data found within plot range/limits for comparison plot '{title}'. Skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Blower Setting")
    ax.set_ylabel("Articulation Index (AI) (%)") # Specific Y label for AI
    ax.set_title(title)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in sorted_settings]) # Use sorted setting labels
    ax.tick_params(axis='x', which='minor', bottom=False)
    if num_settings > 0:
        ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
    else:
        ax.set_xlim(-0.5, 0.5)

    ax.yaxis.grid(True, linestyle=':', linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, zorder=1)
    ax.xaxis.grid(False)
    ax.legend(fontsize='small')

    plt.tight_layout()
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"AI comparison plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving AI comparison plot {output_filename}: {e}")
    plt.close(fig)