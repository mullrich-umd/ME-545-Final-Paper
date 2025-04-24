# main_vehicle_analysis.py
# Application script for analyzing vehicle noise spectrum data

import numpy as np
# import matplotlib.pyplot as plt # Only needed for default colors if used directly
import os
import re
import sys
import json
from collections import defaultdict

# Import the custom library
import audio_spl_lib as spl # Use 'spl' as a prefix for library functions

# --- User Configuration ---
# Define a list of file or directory paths to process automatically.
INPUT_PATHS = [
    "/Users/max/Library/CloudStorage/GoogleDrive-mullrich@umich.edu/My Drive/Classes/W25 - ME 545/Term Paper/Ford Explorer",
    "/Users/max/Library/CloudStorage/GoogleDrive-mullrich@umich.edu/My Drive/Classes/W25 - ME 545/Term Paper/Test-Analysis Procedure - Data Tundra and Odyssey/Honda Odysee", # Corrected typo? "Odyssey"
    "/Users/max/Library/CloudStorage/GoogleDrive-mullrich@umich.edu/My Drive/Classes/W25 - ME 545/Term Paper/Test-Analysis Procedure - Data Tundra and Odyssey/Toyota Tundra"
]

SVG_WIDTH_INCHES = 8
SVG_HEIGHT_INCHES = 5
# This flag now controls the *default* weighting for NB/Octave plots (dBA vs dBZ)
# AI calculation requires dBA (will be skipped if False).
# Loudness calculation requires dBZ (will always be calculated internally).
APPLY_A_WEIGHTING_TO_PLOTS = True

# --- Application Constants ---
# Get the directory where this script is located
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd() # Fallback to current working directory
OFFSET_FILE = os.path.join(SCRIPT_DIR, "gain_offsets.json") # Store offsets next to script

DEFAULT_VEHICLE_NAME = "UnknownVehicle"
DEFAULT_DIR_VEHICLE_NAME = "UnknownVehicleFromDir"
SUBDIR_NARROWBAND = "NarrowBand"
SUBDIR_THIRDOCTAVE = "ThirdOctave"
SUBDIR_COMBINED = "Combined"
SUBDIR_AI = "ArticulationIndex" # Subdir for ALL AI plots (within vehicle dir)
SUBDIR_LOUDNESS = "Loudness" # Subdir for Loudness plots
COMPARISON_SUBDIR = "Comparison_Plots" # Subdirectory for comparison plots
FILENAME_SANITIZE_CHAR = '_' # Character used to replace invalid chars in filenames

# --- Offset Persistence Functions ---
def load_offsets(filepath):
    """Loads vehicle gain offsets from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                offsets = json.load(f)
                if isinstance(offsets, dict):
                    print(f"Loaded gain offsets from: {filepath}")
                    return offsets
                else:
                    print(f"Warning: Invalid format in {filepath}. Starting fresh.")
                    return {}
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}. Starting fresh.")
            return {}
        except Exception as e:
            print(f"Warning: Error loading offsets from {filepath}: {e}. Starting fresh.")
            return {}
    else:
        print("Offset file not found. Will create one if new offsets are entered.")
        return {}

def save_offsets(filepath, offsets):
    """Saves vehicle gain offsets to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(offsets, f, indent=4)
        # print(f"Saved gain offsets to: {filepath}") # Optional confirmation
    except Exception as e:
        print(f"Error: Could not save offsets to {filepath}: {e}")


# --- Function to process a single input path (file or directory) ---
def process_input_path(input_path, gain_offsets_db, band_definitions, vehicle_color_map):
    """
    Processes a single input path (file or directory), calculates SPL, AI, & Loudness,
    generates plots using assigned vehicle colors, and collects data.
    Uses functions from audio_spl_lib.

    Args:
        input_path (str): The file or directory path to process.
        gain_offsets_db (dict): The dictionary of known gain offsets.
        band_definitions (tuple): Pre-calculated 1/3 octave band definitions.
        vehicle_color_map (dict): Map of {vehicle_name: color_string}.

    Returns:
        tuple: (
            dict: The potentially updated gain_offsets_db dictionary,
            dict: {setting_label: octave_spl_array (dBA or dBZ)} for the vehicle, or None.
            dict: {setting_label: (nb_freqs, nb_spl (dBA or dBZ))} for the vehicle, or None.
            dict: {setting_label: ai_value} for the vehicle, or None.
            dict: {setting_label: loudness_value (Sones)} for the vehicle, or None. # Added Loudness
            str: The determined vehicle name, or None.
            str: The base directory used for output, or None.
        )
    """
    filepaths = []
    vehicle_name = DEFAULT_VEHICLE_NAME
    base_dir = None # Initialize base_dir
    is_directory_input = False
    vehicle_octave_data = {}
    vehicle_narrowband_data = {}
    vehicle_ai_data = {}
    vehicle_loudness_data = {} # Added dictionary for loudness

    # Check for A-weighting requirement for AI *before* processing files
    if not APPLY_A_WEIGHTING_TO_PLOTS:
        print("\n" + "="*30)
        print("WARNING: APPLY_A_WEIGHTING_TO_PLOTS is False.")
        print("         Articulation Index (AI) calculation requires A-weighted data")
        print("         and will be skipped.")
        print("         Loudness calculation (using dBZ) will proceed.")
        print("="*30 + "\n")
        # AI calculation will be skipped later, no need to exit here

    if not os.path.exists(input_path):
        print(f"Error: Path not found - {input_path}")
        return gain_offsets_db, None, None, None, None, None, None # Added None for loudness

    if os.path.isfile(input_path):
        if input_path.lower().endswith('.txt'):
            filepaths.append(input_path)
            parent_dir = os.path.dirname(input_path)
            if parent_dir:
                 abs_parent_dir = os.path.abspath(parent_dir)
                 potential_name = os.path.basename(abs_parent_dir)
                 # Try to get vehicle name from grandparent if parent is a standard subdir
                 standard_subdirs = {SUBDIR_NARROWBAND, SUBDIR_THIRDOCTAVE, SUBDIR_COMBINED, SUBDIR_AI, SUBDIR_LOUDNESS, COMPARISON_SUBDIR} # Added Loudness subdir
                 if potential_name in standard_subdirs:
                     grandparent_dir = os.path.dirname(abs_parent_dir)
                     potential_name_gp = os.path.basename(grandparent_dir)
                     if potential_name_gp and potential_name_gp != os.path.sep:
                         vehicle_name = potential_name_gp
                         base_dir = grandparent_dir # Output goes to vehicle dir
                     else: # Fallback if grandparent is root or invalid
                         vehicle_name = DEFAULT_VEHICLE_NAME
                         base_dir = abs_parent_dir # Output goes to parent dir
                 else: # Parent is likely the vehicle dir
                     vehicle_name = potential_name if potential_name and potential_name != os.path.sep else DEFAULT_VEHICLE_NAME
                     base_dir = abs_parent_dir # Output goes to parent dir
            else: # File is in the root or script dir
                 vehicle_name = DEFAULT_VEHICLE_NAME
                 base_dir = os.path.dirname(os.path.abspath(input_path)) # Output goes to file's dir

            is_directory_input = False
            print(f"Processing single file: {input_path}. Vehicle name assumed: '{vehicle_name}'. Output in: {base_dir}")
        else:
            print(f"Error: Input file must be a .txt file. Provided: {input_path}")
            return gain_offsets_db, None, None, None, None, None, None # Added None for loudness

    elif os.path.isdir(input_path):
        is_directory_input = True
        abs_input_path = os.path.abspath(input_path)
        vehicle_name = os.path.basename(abs_input_path)
        if not vehicle_name or vehicle_name == os.path.sep:
            vehicle_name = DEFAULT_DIR_VEHICLE_NAME
        base_dir = abs_input_path # Output goes into this directory
        print(f"Processing directory: {input_path}. Vehicle name: '{vehicle_name}'. Output in: {base_dir}")

        found_txt = False
        for filename in os.listdir(abs_input_path):
            if filename.lower().endswith('.txt'):
                full_path = os.path.join(abs_input_path, filename)
                if os.path.isfile(full_path):
                    filepaths.append(full_path)
                    found_txt = True

        if not found_txt:
            print(f"No .txt files found in directory: {input_path}")
            return gain_offsets_db, None, None, None, None, None, None # Added None for loudness

        try:
            # Use the sort key function from the library
            filepaths.sort(key=lambda f: spl.get_setting_sort_key(
                (re.findall(r'(\d+|OFF)\b', os.path.basename(f), re.IGNORECASE) or [os.path.basename(f)])[-1]
            ))
        except Exception as e:
            print(f"Warning: Could not reliably sort input files based on setting label ({e}). Using default sort.")
            filepaths.sort()

    else:
        print(f"Error: Input path is neither a file nor a directory - {input_path}")
        return gain_offsets_db, None, None, None, None, None, None # Added None for loudness

    if not filepaths:
        print(f"No valid .txt files to process for path: {input_path}")
        return gain_offsets_db, None, None, None, None, None, None # Added None for loudness

    # --- Get Gain Offset ---
    gain_offset = None
    if vehicle_name in gain_offsets_db:
        gain_offset = gain_offsets_db[vehicle_name]
        print(f"Using stored Gain Offset for '{vehicle_name}': {gain_offset:.2f} dB")
    else:
        print(f"No stored offset found for vehicle '{vehicle_name}'.")
        while True:
            try:
                offset_str = input(f"Enter the Gain Offset (dB) for vehicle '{vehicle_name}': ")
                gain_offset = float(offset_str)
                gain_offsets_db[vehicle_name] = gain_offset
                save_offsets(OFFSET_FILE, gain_offsets_db) # Save immediately
                print(f"Stored new offset for '{vehicle_name}'.")
                break
            except ValueError:
                print("Invalid input. Please enter a number for the gain offset.")
            except EOFError:
                print("\nInput stream closed. Cannot prompt for offset. Exiting.")
                sys.exit(1)
    # --- End Gain Offset Handling ---

    # Get assigned vehicle color (use first default color as fallback)
    # Access default colors via the library
    vehicle_color = vehicle_color_map.get(vehicle_name, spl.DEFAULT_COLORS[0])

    all_narrowband_data_for_combined_plot = []

    # Process each file
    for filepath in filepaths:
        print(f"\nProcessing file: {filepath}")

        filename = os.path.basename(filepath)
        matches = re.findall(r'(\d+|OFF)\b', filename, re.IGNORECASE)
        setting_label = None
        if matches:
            setting_label = matches[-1].upper()
            if setting_label == "OFF":
                setting_label = "Off"
        else:
            print(f"Warning: Could not extract setting number/label from '{filename}'. Using filename base.")
            setting_label = os.path.splitext(filename)[0]

        # Use library function to read data
        nb_freqs, nb_audacity_db = spl.read_spectrum_data(filepath)

        if nb_freqs is not None and nb_audacity_db is not None:
            # --- SPL Calculations ---
            # Use library function: returns (final_spl, unweighted_spl)
            # final_spl is dBA or dBZ based on APPLY_A_WEIGHTING_TO_PLOTS
            # unweighted_spl is always dBZ
            nb_spl_final, nb_spl_z = spl.calculate_narrowband_spl(
                nb_audacity_db, gain_offset, nb_freqs, APPLY_A_WEIGHTING_TO_PLOTS
            )

            # Calculate 1/3 Octave SPL using the 'final' SPL (dBA or dBZ)
            octave_spl_final = spl.calculate_third_octave_spl(nb_freqs, nb_spl_final, band_definitions)

            # --- Articulation Index Calculation (Requires dBA) ---
            ai_value = np.nan # Default to NaN
            ai_details = None
            if APPLY_A_WEIGHTING_TO_PLOTS:
                # If plots are A-weighted, octave_spl_final is already dBA
                octave_spl_a = octave_spl_final
                ai_details = spl.get_ai_calculation_details(spl.NOMINAL_FREQUENCIES, octave_spl_a)
                ai_value = ai_details['total_ai']
                print(f"  Calculated Articulation Index (AI): {ai_value:.2f}%")
            else:
                # Need to calculate A-weighted octave SPL separately for AI
                print("  Calculating A-weighted SPL specifically for AI...")
                # Recalculate NB SPL as dBA (ignore first return value from calc_narrowband_spl)
                _, nb_spl_a_for_ai = spl.calculate_narrowband_spl(
                    nb_audacity_db, gain_offset, nb_freqs, apply_a_weighting=True
                )
                octave_spl_a = spl.calculate_third_octave_spl(nb_freqs, nb_spl_a_for_ai, band_definitions)
                ai_details = spl.get_ai_calculation_details(spl.NOMINAL_FREQUENCIES, octave_spl_a)
                ai_value = ai_details['total_ai']
                print(f"  Calculated Articulation Index (AI): {ai_value:.2f}%")
                # Note: The main octave_spl_final variable remains dBZ for plotting if APPLY_A_WEIGHTING_TO_PLOTS is False

            # --- Loudness Calculation (Requires dBZ) ---
            loudness_value_sone = np.nan # Default to NaN
            if spl.MOSQITO_AVAILABLE:
                print("  Calculating Loudness (Zwicker)...")
                # Convert dBZ narrow-band SPL to RMS pressure
                rms_pressure = spl.spl_dbz_to_rms_pressure(nb_spl_z, nb_freqs)

                # Filter out non-positive frequencies and corresponding pressures for MOSQITO
                valid_loudness_indices = nb_freqs > 0
                if np.any(valid_loudness_indices):
                    freqs_for_loudness = nb_freqs[valid_loudness_indices]
                    rms_pressure_for_loudness = rms_pressure[valid_loudness_indices]

                    # --- BEGIN DEBUG BLOCK ---
                    # Check the actual frequency range being passed to MOSQITO
                    min_freq_passed = np.min(freqs_for_loudness)
                    max_freq_passed = np.max(freqs_for_loudness)
                    # print(f"  DEBUG: Frequencies passed to MOSQITO - Min: {min_freq_passed:.2f} Hz, Max: {max_freq_passed:.2f} Hz")
                    # print(f"  DEBUG: Number of frequency points passed: {len(freqs_for_loudness)}")

                    # Check if the range meets the standard's ideal minimum/maximum
                    # if min_freq_passed > 24:
                    #     print(f"  DEBUG: Minimum frequency {min_freq_passed:.2f} Hz is > 24 Hz. Low frequencies might be zero-filled by MOSQITO.")
                    # if max_freq_passed < 24000:
                    #     print(f"  DEBUG: Maximum frequency {max_freq_passed:.2f} Hz is < 24 kHz. High frequencies might be zero-filled by MOSQITO.")
                    # --- END DEBUG BLOCK ---


                    # Ensure spectrum is not all zeros
                    if np.any(rms_pressure_for_loudness > spl.SMALL_EPSILON):
                        try:
                            # Call MOSQITO function (field_type='free' is default)
                            N, N_spec, bark_axis = spl.loudness_zwst_freq(
                                rms_pressure_for_loudness, freqs_for_loudness
                            )
                            loudness_value_sone = N
                            print(f"  Calculated Loudness: {loudness_value_sone:.2f} Sones")
                        except Exception as e:
                            print(f"  Error during MOSQITO loudness calculation: {e}")
                    else:
                        print("  Skipping loudness calculation: RMS pressure spectrum is effectively zero.")
                        loudness_value_sone = 0.0 # Treat zero pressure as zero loudness
                else:
                    print("  Skipping loudness calculation: No positive frequencies found.")
            else:
                print("  Skipping loudness calculation: MOSQITO library not available.")


            # Store data for comparisons/combined plots
            if setting_label is not None:
                vehicle_octave_data[setting_label] = octave_spl_final # Store dBA or dBZ based on flag
                vehicle_ai_data[setting_label] = ai_value # Store AI value (always dBA based)
                vehicle_loudness_data[setting_label] = loudness_value_sone # Store Loudness value

                if is_directory_input: # Only store NB data if processing a directory
                    vehicle_narrowband_data[setting_label] = (nb_freqs, nb_spl_final) # Store dBA or dBZ
                    all_narrowband_data_for_combined_plot.append(
                        (setting_label, nb_freqs, nb_spl_final) # Use final SPL for combined plot
                    )

            # Create titles and filenames
            # Suffix reflects the primary weighting used for NB/Octave plots
            weighting_suffix_plots = "A" if APPLY_A_WEIGHTING_TO_PLOTS else "Z"
            safe_vehicle_name = re.sub(r'[\\/*?:"<>|]+', FILENAME_SANITIZE_CHAR, vehicle_name)
            safe_setting_label = re.sub(r'[\\/*?:"<>|]+', FILENAME_SANITIZE_CHAR, str(setting_label))
            base_title = f"{safe_vehicle_name} Blower Setting {safe_setting_label}"
            base_filename = f"{safe_vehicle_name}_Setting_{safe_setting_label}"

            # Define Full Output Paths using application constants and base_dir
            nb_output_dir = os.path.join(base_dir, SUBDIR_NARROWBAND)
            oct_output_dir = os.path.join(base_dir, SUBDIR_THIRDOCTAVE)
            ai_output_dir = os.path.join(base_dir, SUBDIR_AI) # AI plots go here
            loudness_output_dir = os.path.join(base_dir, SUBDIR_LOUDNESS) # Loudness plots go here

            nb_output_svg = os.path.join(nb_output_dir, f"{base_filename}_NarrowBand_SPL{weighting_suffix_plots}.svg")
            oct_output_svg = os.path.join(oct_output_dir, f"{base_filename}_ThirdOctave_SPL{weighting_suffix_plots}.svg")
            # AI detail plot always uses dBA, so suffix is 'A' regardless of main flag
            ai_detail_output_svg = os.path.join(ai_output_dir, f"{base_filename}_AIDetail_SPL_A.svg")
            # Loudness plot has its own units (Sones)
            # loudness_detail_output_svg = os.path.join(loudness_output_dir, f"{base_filename}_Loudness.svg") # Individual Loudness plot (not generated here)

            # --- Plotting ---
            # Use library functions for plotting individual results

            # Narrowband Plot (dBA or dBZ based on flag)
            nb_plot_title = f"{base_title} - Narrow Band SPL ({'dBA' if APPLY_A_WEIGHTING_TO_PLOTS else 'dBZ'})"
            spl.plot_narrowband_spl(
                nb_freqs, nb_spl_final, nb_plot_title, nb_output_svg,
                SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES, APPLY_A_WEIGHTING_TO_PLOTS,
                color=vehicle_color
            )

            # 1/3 Octave Plot (dBA or dBZ based on flag)
            oct_plot_title = f"{base_title} - 1/3 Octave Band SPL ({'dBA' if APPLY_A_WEIGHTING_TO_PLOTS else 'dBZ'})"
            spl.plot_third_octave_spl(
                spl.NOMINAL_FREQUENCIES, octave_spl_final, oct_plot_title, oct_output_svg,
                SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES, APPLY_A_WEIGHTING_TO_PLOTS,
                color=vehicle_color
            )

            # AI Detail Plot (Only if AI was calculated)
            if ai_details is not None: # Check if details exist (implies AI was calculated)
                ai_detail_title = f"{base_title} - AI Band SPL & Limits"
                spl.plot_ai_band_details(
                    ai_details, # Pass the detailed results
                    ai_detail_title,
                    ai_detail_output_svg,
                    SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES,
                    color=vehicle_color
                )
            elif not APPLY_A_WEIGHTING_TO_PLOTS:
                 print("  Skipping AI Detail plot generation (APPLY_A_WEIGHTING_TO_PLOTS is False).")
            # No else needed if ai_details is None, means calculation failed earlier

            # Loudness Plot (vs Setting) - This plot is generated later for the whole vehicle

            # --- End Plotting ---

        else:
            print(f"Skipping calculations and plots for {filepath} due to read error.")

    # Generate Combined Narrow-Band Plot (only if input was a directory)
    if is_directory_input and len(all_narrowband_data_for_combined_plot) > 1:
        print(f"\nGenerating combined narrow-band plot for '{vehicle_name}'...")
        weighting_suffix_plots = "A" if APPLY_A_WEIGHTING_TO_PLOTS else "Z"
        safe_vehicle_name = re.sub(r'[\\/*?:"<>|]+', FILENAME_SANITIZE_CHAR, vehicle_name)
        combined_output_dir = os.path.join(base_dir, SUBDIR_COMBINED)
        combined_title = f"{safe_vehicle_name} All Settings - Combined Narrow Band SPL ({'dBA' if APPLY_A_WEIGHTING_TO_PLOTS else 'dBZ'})"
        combined_filename = os.path.join(combined_output_dir, f"{safe_vehicle_name}_AllSettings_Combined_NarrowBand_SPL{weighting_suffix_plots}.svg")

        # Use library function for combined plot
        spl.plot_combined_narrowband_spl(
            all_narrowband_data_for_combined_plot,
            combined_title, combined_filename,
            SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES, APPLY_A_WEIGHTING_TO_PLOTS
        )
    elif is_directory_input and len(all_narrowband_data_for_combined_plot) <= 1:
         print(f"\nSkipping combined narrow-band plot for '{vehicle_name}' (only one or zero valid datasets found).")

    print(f"\nFinished processing for path: {input_path}")
    # Return data only if processing a directory successfully
    returned_octave_data = vehicle_octave_data if is_directory_input and vehicle_octave_data else None
    returned_narrowband_data = vehicle_narrowband_data if is_directory_input and vehicle_narrowband_data else None
    returned_ai_data = vehicle_ai_data if is_directory_input and vehicle_ai_data else None
    returned_loudness_data = vehicle_loudness_data if is_directory_input and vehicle_loudness_data else None # Added loudness
    returned_vehicle_name = vehicle_name if is_directory_input else None
    returned_base_dir = base_dir if is_directory_input else None # Return base_dir only for directories

    return gain_offsets_db, returned_octave_data, returned_narrowband_data, returned_ai_data, returned_loudness_data, returned_vehicle_name, returned_base_dir


# --- Main Execution ---
if __name__ == "__main__":

    # Use library function to get band definitions
    band_definitions = spl.get_third_octave_bands(spl.NOMINAL_FREQUENCIES)

    # Load existing offsets using local function
    gain_offsets_db = load_offsets(OFFSET_FILE)

    all_vehicles_octave_data = {}
    all_vehicles_narrowband_data = {}
    all_vehicles_ai_data = {}
    all_vehicles_loudness_data = {} # Added loudness data store
    vehicle_color_map = {}
    vehicle_base_dir_map = {} # Store base output directory for each vehicle

    if INPUT_PATHS:
        print("Processing paths defined in INPUT_PATHS list...")

        # Pre-determine unique vehicle names and assign colors
        unique_vehicle_names = set()
        temp_base_dir_map = {} # Temporary map during initial scan
        for path in INPUT_PATHS:
            vehicle_name = None
            base_dir = None
            if os.path.isdir(path):
                abs_path = os.path.abspath(path)
                vehicle_name = os.path.basename(abs_path)
                if not vehicle_name or vehicle_name == os.path.sep:
                    vehicle_name = DEFAULT_DIR_VEHICLE_NAME
                base_dir = abs_path
            elif os.path.isfile(path) and path.lower().endswith('.txt'):
                 # Try to determine vehicle name and base dir for single files
                 parent_dir = os.path.dirname(path)
                 if parent_dir:
                     abs_parent_dir = os.path.abspath(parent_dir)
                     potential_name = os.path.basename(abs_parent_dir)
                     standard_subdirs = {SUBDIR_NARROWBAND, SUBDIR_THIRDOCTAVE, SUBDIR_COMBINED, SUBDIR_AI, SUBDIR_LOUDNESS, COMPARISON_SUBDIR} # Added Loudness
                     if potential_name in standard_subdirs:
                         grandparent_dir = os.path.dirname(abs_parent_dir)
                         potential_name_gp = os.path.basename(grandparent_dir)
                         if potential_name_gp and potential_name_gp != os.path.sep:
                             vehicle_name = potential_name_gp
                             base_dir = grandparent_dir
                         else:
                             vehicle_name = DEFAULT_VEHICLE_NAME
                             base_dir = abs_parent_dir
                     else:
                         vehicle_name = potential_name if potential_name and potential_name != os.path.sep else DEFAULT_VEHICLE_NAME
                         base_dir = abs_parent_dir
                 else:
                     vehicle_name = DEFAULT_VEHICLE_NAME
                     base_dir = os.path.dirname(os.path.abspath(path))

            if vehicle_name and base_dir:
                unique_vehicle_names.add(vehicle_name)
                # Store the first base_dir found for a vehicle name
                if vehicle_name not in temp_base_dir_map:
                    temp_base_dir_map[vehicle_name] = base_dir


        sorted_vehicle_names = sorted(list(unique_vehicle_names))
        # Access default colors via the library
        num_colors = len(spl.DEFAULT_COLORS)
        for i, name in enumerate(sorted_vehicle_names):
            vehicle_color_map[name] = spl.DEFAULT_COLORS[i % num_colors]
            print(f"Assigned color {vehicle_color_map[name]} to vehicle '{name}'")
            # Assign the determined base directory
            if name in temp_base_dir_map:
                vehicle_base_dir_map[name] = temp_base_dir_map[name]
                print(f"  Output base directory for '{name}': {vehicle_base_dir_map[name]}")
            else:
                 print(f"  Warning: Could not determine base directory for '{name}' during initial scan.")
                 # Assign a default or handle later if needed
                 vehicle_base_dir_map[name] = os.path.join(SCRIPT_DIR, name) # Default to script dir + name


        # Process each path
        for path in INPUT_PATHS:
            print(f"\n--- Processing Input: {path} ---")
            # Call local processing function (now returns loudness data and base_dir)
            gain_offsets_db, vehicle_octave_data, vehicle_narrowband_data, vehicle_ai_data, vehicle_loudness_data, vehicle_name, base_dir = process_input_path(
                path, gain_offsets_db, band_definitions, vehicle_color_map
            )
            # Store results if valid (process_input_path now returns None for data if not directory)
            if vehicle_name and vehicle_octave_data:
                 all_vehicles_octave_data[vehicle_name] = vehicle_octave_data
            if vehicle_name and vehicle_narrowband_data:
                all_vehicles_narrowband_data[vehicle_name] = vehicle_narrowband_data
            if vehicle_name and vehicle_ai_data:
                all_vehicles_ai_data[vehicle_name] = vehicle_ai_data
            if vehicle_name and vehicle_loudness_data: # Store loudness data
                all_vehicles_loudness_data[vehicle_name] = vehicle_loudness_data

            # Update base directory map if processing was successful for a directory
            if vehicle_name and base_dir and (vehicle_octave_data or vehicle_narrowband_data or vehicle_ai_data or vehicle_loudness_data):
                if vehicle_name not in vehicle_base_dir_map:
                     vehicle_base_dir_map[vehicle_name] = base_dir
                     print(f"  Stored output base directory for '{vehicle_name}': {base_dir}")
                elif vehicle_base_dir_map[vehicle_name] != base_dir:
                     # This might happen if multiple INPUT_PATHS resolve to the same vehicle name
                     # but have different base directories. We'll use the one from the last successful processing.
                     print(f"  Updating base directory for '{vehicle_name}' to: {base_dir}")
                     vehicle_base_dir_map[vehicle_name] = base_dir

            elif vehicle_name is None and vehicle_octave_data is None and vehicle_narrowband_data is None and vehicle_ai_data is None and vehicle_loudness_data is None and base_dir is None:
                pass # Invalid path or single file processing
            # else: # This case should be less likely now
            #     print(f"Warning: Inconsistent return from process_input_path for {path}")


        print("\n--- Finished processing all paths defined in INPUT_PATHS ---")


        # --- Generate AI vs Setting Plots ---
        if all_vehicles_ai_data:
            print(f"\n--- Generating Articulation Index vs. Setting Plots ---")

            for vehicle_name, ai_data in all_vehicles_ai_data.items():
                if ai_data: # Check if there's actually AI data for this vehicle
                    if vehicle_name not in vehicle_base_dir_map:
                        print(f"  Warning: Cannot determine output directory for AI plot of '{vehicle_name}'. Skipping.")
                        continue

                    print(f"  Generating AI vs Setting plot for '{vehicle_name}'...")
                    safe_vehicle_name = re.sub(r'[\\/*?:"<>|]+', FILENAME_SANITIZE_CHAR, vehicle_name)
                    # Save AI plots in the vehicle-specific AI subfolder
                    vehicle_base_dir = vehicle_base_dir_map[vehicle_name]
                    vehicle_ai_output_dir = os.path.join(vehicle_base_dir, SUBDIR_AI) # Use vehicle's base dir
                    ai_plot_filename = os.path.join(vehicle_ai_output_dir, f"{safe_vehicle_name}_ArticulationIndex_vs_Setting.svg")
                    vehicle_color = vehicle_color_map.get(vehicle_name, spl.DEFAULT_COLORS[0]) # Get assigned color

                    # Use the plotting function from the library
                    spl.plot_articulation_index_by_setting(
                        vehicle_name,
                        ai_data,
                        ai_plot_filename,
                        SVG_WIDTH_INCHES, # Use existing width/height config
                        SVG_HEIGHT_INCHES,
                        color=vehicle_color
                    )
                else:
                    print(f"  Skipping AI vs Setting plot for '{vehicle_name}' (no AI data calculated or stored).")
            print("Finished generating AI vs Setting plots.")
        else:
            print("\nNo Articulation Index data was calculated/stored for any vehicle. Skipping AI vs Setting plots.")
        # --- End AI vs Setting Plot Generation ---


        # --- Generate Loudness vs Setting Plots ---
        if all_vehicles_loudness_data:
            print(f"\n--- Generating Loudness vs. Setting Plots ---")

            for vehicle_name, loudness_data in all_vehicles_loudness_data.items():
                if loudness_data: # Check if there's actually Loudness data for this vehicle
                    if vehicle_name not in vehicle_base_dir_map:
                        print(f"  Warning: Cannot determine output directory for Loudness plot of '{vehicle_name}'. Skipping.")
                        continue

                    print(f"  Generating Loudness vs Setting plot for '{vehicle_name}'...")
                    safe_vehicle_name = re.sub(r'[\\/*?:"<>|]+', FILENAME_SANITIZE_CHAR, vehicle_name)
                    # Save Loudness plots in the vehicle-specific Loudness subfolder
                    vehicle_base_dir = vehicle_base_dir_map[vehicle_name]
                    vehicle_loudness_output_dir = os.path.join(vehicle_base_dir, SUBDIR_LOUDNESS) # Use vehicle's base dir
                    loudness_plot_filename = os.path.join(vehicle_loudness_output_dir, f"{safe_vehicle_name}_Loudness_vs_Setting.svg")
                    vehicle_color = vehicle_color_map.get(vehicle_name, spl.DEFAULT_COLORS[0]) # Get assigned color

                    # Use the NEW plotting function from the library
                    spl.plot_loudness_by_setting(
                        vehicle_name,
                        loudness_data,
                        loudness_plot_filename,
                        SVG_WIDTH_INCHES, # Use existing width/height config
                        SVG_HEIGHT_INCHES,
                        color=vehicle_color
                    )
                else:
                    print(f"  Skipping Loudness vs Setting plot for '{vehicle_name}' (no Loudness data calculated or stored).")
            print("Finished generating Loudness vs Setting plots.")
        else:
            print("\nNo Loudness data was calculated/stored for any vehicle. Skipping Loudness vs Setting plots.")
        # --- End Loudness vs Setting Plot Generation ---


        # --- Generate Comparison Plots ---
        # Check based on octave data, but loudness comparison relies on loudness data
        num_vehicles_with_octave = len(all_vehicles_octave_data)
        num_vehicles_with_loudness = len(all_vehicles_loudness_data)
        num_vehicles_with_ai = len(all_vehicles_ai_data)

        # Proceed if we have data for more than one vehicle for *any* comparison type
        if num_vehicles_with_octave > 1 or num_vehicles_with_loudness > 1 or num_vehicles_with_ai > 1: # Add AI check
            print(f"\n--- Generating Comparison Plots ---")
            print(f"  (Found {num_vehicles_with_octave} vehicles with Octave data)")
            print(f"  (Found {num_vehicles_with_loudness} vehicles with Loudness data)")
            print(f"  (Found {num_vehicles_with_ai} vehicles with AI data)") # Add AI count print

            # Comparison plots go in a top-level directory relative to the script
            comparison_output_dir = os.path.join(SCRIPT_DIR, COMPARISON_SUBDIR)
            os.makedirs(comparison_output_dir, exist_ok=True)
            # Suffix reflects the primary weighting used for NB/Octave plots
            weighting_suffix_plots = "A" if APPLY_A_WEIGHTING_TO_PLOTS else "Z"

            # --- Plot 1/3 Octave and Narrowband Comparisons ---
            if num_vehicles_with_octave > 1:
                # Find common settings based on octave data
                settings_per_vehicle_octave = [
                    set(data.keys()) for data in all_vehicles_octave_data.values() if data
                ]
                common_settings_octave = set()
                if settings_per_vehicle_octave:
                    try:
                        valid_keys_sets_octave = [
                            set(k for k in s if k is not None) for s in settings_per_vehicle_octave
                        ]
                        if valid_keys_sets_octave:
                            common_settings_octave = set.intersection(*valid_keys_sets_octave)
                    except TypeError:
                        print("Warning: Could not compute intersection of settings for Octave/NB comparison.")
                        common_settings_octave = set()

                if common_settings_octave:
                    sorted_common_settings_octave = sorted(list(common_settings_octave), key=spl.get_setting_sort_key)
                    print(f"\nCommon settings for NB/Octave plots: {', '.join(map(str, sorted_common_settings_octave))}")
                    print(f"Generating NB/Octave comparison plots for {len(sorted_common_settings_octave)} common settings...")
                    for setting in sorted_common_settings_octave:
                        print(f"  Generating NB/Octave comparison plots for Setting '{setting}'...")
                        safe_setting = re.sub(r'[\\/*?:"<>|]+', FILENAME_SANITIZE_CHAR, str(setting))

                        # --- 1/3 Octave Comparison Plot ---
                        current_setting_octave_data = {
                            v_name: data[setting]
                            for v_name, data in all_vehicles_octave_data.items()
                            if data and setting in data and np.isfinite(data[setting]).any() # Check data exists and has finite values
                        }
                        if current_setting_octave_data:
                            comp_title = f"Vehicle Comparison - Setting '{setting}' - 1/3 Octave SPL ({'dBA' if APPLY_A_WEIGHTING_TO_PLOTS else 'dBZ'})"
                            comp_filename = os.path.join(comparison_output_dir, f"Comparison_Setting_{safe_setting}_ThirdOctave_SPL{weighting_suffix_plots}.svg")
                            spl.plot_grouped_third_octave_spl(
                                current_setting_octave_data,
                                spl.NOMINAL_FREQUENCIES,
                                comp_title, comp_filename,
                                SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES, APPLY_A_WEIGHTING_TO_PLOTS,
                                vehicle_color_map
                            )
                        else:
                            print(f"    Warning: No valid 1/3 octave data found for setting '{setting}'. Skipping 1/3 octave comparison plot.")

                        # --- Narrowband Comparison Plot ---
                        current_setting_narrowband_data = {
                            v_name: nb_data[setting]
                            for v_name, nb_data in all_vehicles_narrowband_data.items()
                            if nb_data and setting in nb_data and nb_data[setting] is not None and len(nb_data[setting]) == 2
                        }
                        current_setting_narrowband_data_filtered = {
                            v_name: data
                            for v_name, data in current_setting_narrowband_data.items()
                            if np.isfinite(data[1]).any()
                        }
                        if current_setting_narrowband_data_filtered:
                            nb_comp_title = f"Vehicle Comparison - Setting '{setting}' - Narrow Band SPL ({'dBA' if APPLY_A_WEIGHTING_TO_PLOTS else 'dBZ'})"
                            nb_comp_filename = os.path.join(comparison_output_dir, f"Comparison_Setting_{safe_setting}_NarrowBand_SPL{weighting_suffix_plots}.svg")
                            spl.plot_comparison_narrowband_spl(
                                current_setting_narrowband_data_filtered,
                                nb_comp_title, nb_comp_filename,
                                SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES, APPLY_A_WEIGHTING_TO_PLOTS,
                                vehicle_color_map
                            )
                        else:
                             print(f"    Warning: No valid narrowband data found for setting '{setting}'. Skipping narrowband comparison plot.")
                else:
                    print("\nNo common settings found for NB/Octave data. Skipping those comparison plots.")
            else:
                print("\nFewer than two vehicles have Octave/NB data. Skipping those comparison plots.")


            # --- Loudness Comparison Plot ---
            if num_vehicles_with_loudness > 1:
                # Find common settings based on *loudness* data specifically
                settings_per_vehicle_loudness = [
                    set(data.keys()) for data in all_vehicles_loudness_data.values() if data
                ]
                common_settings_loudness = set()
                if settings_per_vehicle_loudness:
                    try:
                        valid_keys_sets_loudness = [
                            set(k for k in s if k is not None) for s in settings_per_vehicle_loudness
                        ]
                        if valid_keys_sets_loudness:
                            # Also check for finite values before intersection
                            valid_keys_sets_loudness_finite = []
                            for v_name, settings_set in zip(all_vehicles_loudness_data.keys(), valid_keys_sets_loudness):
                                if all_vehicles_loudness_data[v_name]: # Check if dict is not empty
                                    finite_set = {
                                        k for k in settings_set
                                        if k in all_vehicles_loudness_data[v_name] and np.isfinite(all_vehicles_loudness_data[v_name][k])
                                    }
                                    valid_keys_sets_loudness_finite.append(finite_set)

                            if valid_keys_sets_loudness_finite:
                                common_settings_loudness = set.intersection(*valid_keys_sets_loudness_finite)

                    except TypeError:
                        print("Warning: Could not compute intersection of settings for Loudness comparison.")
                        common_settings_loudness = set()

                if common_settings_loudness:
                    print(f"\nGenerating Loudness comparison plot...")
                    loudness_comp_title = f"Vehicle Comparison - Loudness vs. Setting"
                    loudness_comp_filename = os.path.join(comparison_output_dir, f"Comparison_AllSettings_Loudness.svg")

                    # Filter data passed to plot function to only include vehicles with *some* loudness data
                    valid_loudness_data_for_comp = {
                        v_name: data
                        for v_name, data in all_vehicles_loudness_data.items()
                        if data # Ensure the vehicle has a loudness dictionary
                    }

                    if len(valid_loudness_data_for_comp) > 1:
                        # Call the new library function
                        spl.plot_comparison_loudness_by_setting(
                            valid_loudness_data_for_comp,
                            loudness_comp_title,
                            loudness_comp_filename,
                            SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES,
                            vehicle_color_map
                        )
                    else:
                        # This check might be redundant given num_vehicles_with_loudness > 1, but safe
                        print("  Warning: Fewer than two vehicles have valid loudness data after filtering. Skipping Loudness comparison plot.")
                else:
                    print("\nNo common settings with valid Loudness data found across vehicles. Skipping Loudness comparison plot.")
            else:
                print("\nFewer than two vehicles have Loudness data. Skipping Loudness comparison plot.")
            # --- End Loudness Comparison ---

            # --- Articulation Index (AI) Comparison Plot ---
            if num_vehicles_with_ai > 1:
                # Find common settings based on *AI* data specifically
                settings_per_vehicle_ai = [
                    set(data.keys()) for data in all_vehicles_ai_data.values() if data
                ]
                common_settings_ai = set()
                if settings_per_vehicle_ai:
                    try:
                        valid_keys_sets_ai = [
                            set(k for k in s if k is not None) for s in settings_per_vehicle_ai
                        ]
                        if valid_keys_sets_ai:
                            # Also check for finite values before intersection
                            valid_keys_sets_ai_finite = []
                            for v_name, settings_set in zip(all_vehicles_ai_data.keys(), valid_keys_sets_ai):
                                if all_vehicles_ai_data[v_name]: # Check if dict is not empty
                                    finite_set = {
                                        k for k in settings_set
                                        if k in all_vehicles_ai_data[v_name] and np.isfinite(all_vehicles_ai_data[v_name][k])
                                    }
                                    valid_keys_sets_ai_finite.append(finite_set)

                            if valid_keys_sets_ai_finite:
                                common_settings_ai = set.intersection(*valid_keys_sets_ai_finite)

                    except TypeError:
                        print("Warning: Could not compute intersection of settings for AI comparison.")
                        common_settings_ai = set()

                if common_settings_ai:
                    print(f"\nGenerating Articulation Index (AI) comparison plot...")
                    ai_comp_title = f"Vehicle Comparison - Articulation Index (AI) vs. Setting"
                    ai_comp_filename = os.path.join(comparison_output_dir, f"Comparison_AllSettings_AI.svg")

                    # Filter data passed to plot function to only include vehicles with *some* AI data
                    valid_ai_data_for_comp = {
                        v_name: data
                        for v_name, data in all_vehicles_ai_data.items()
                        if data # Ensure the vehicle has an AI dictionary
                    }

                    if len(valid_ai_data_for_comp) > 1:
                        # Call the new library function
                        spl.plot_comparison_ai_by_setting( # Call the new function
                            valid_ai_data_for_comp,
                            ai_comp_title,
                            ai_comp_filename,
                            SVG_WIDTH_INCHES, SVG_HEIGHT_INCHES,
                            vehicle_color_map
                            # Uses default y_min/y_max from the function definition
                        )
                    else:
                        # This check might be redundant given num_vehicles_with_ai > 1, but safe
                        print("  Warning: Fewer than two vehicles have valid AI data after filtering. Skipping AI comparison plot.")
                else:
                    print("\nNo common settings with valid AI data found across vehicles. Skipping AI comparison plot.")
            else:
                print("\nFewer than two vehicles have AI data. Skipping AI comparison plot.")
            # --- End AI Comparison ---

            print("\nFinished generating comparison plots.")

        else: # Fewer than 2 vehicles processed overall
             print("\nFewer than two vehicles successfully processed or have data. Skipping comparison plots.")

    # Fallback to interactive mode
    else:
        print("INPUT_PATHS list is empty. Falling back to interactive mode.")
        try:
            input_path_str = input("Enter a single .txt file path OR a directory path containing .txt files:\n")
            input_path = input_path_str.strip().strip('\'"')
            # Call local processing function, ignore comparison data return
            # Interactive mode doesn't generate overall AI/Loudness plots or comparisons
            gain_offsets_db, _, _, _, _, _, _ = process_input_path( # Added _ for loudness
                input_path, gain_offsets_db, band_definitions, {} # Pass empty color map
            )
        except EOFError:
             print("\nInput stream closed. Exiting.")
             sys.exit(1)
        except Exception as e:
             print(f"\nAn unexpected error occurred during interactive input: {e}")
             sys.exit(1)

    print("\nScript finished.")