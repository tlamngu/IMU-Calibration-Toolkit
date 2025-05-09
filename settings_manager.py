# settings_manager.py
import json
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "app_version": "1.5-Qt-HybridAHRS-TuneAll-Logging", # Updated version
    "calib_file": "sensor_calib_ekf.json",
    "max_plot_points": 1000,
    "mag_calib_samples": 1000,
    "accel_gyro_calib_samples": 500,
    "default_baudrate": 115200,
    "gravity": 9.80665,
    "gui_update_interval_ms": 50, # Note: GUI specific, less relevant for pure backend
    "plot_update_interval_ms": 100, # Note: Plot specific, less relevant for pure backend
    "min_valid_dt_sec": 0.0005,
    "max_valid_dt_sec": 0.5,
    "benchmark_duration_sec": 30,
    "max_benchmark_history": 10,

    # AHRS Settings
    "ahrs_algorithm": "EKF",  # Options: "EKF", "Madgwick"
    "madgwick_beta": 0.1,     # Gain for Madgwick algorithm

    # EKF Settings
    # "use_ekf_filter": True, # Deprecated by ahrs_algorithm
    "ekf_q_gyro_noise": 0.001,
    "ekf_q_gyro_bias_noise": 0.00001,
    "ekf_r_accel_noise": 0.1,
    "ekf_r_mag_noise": 0.5,
    "ekf_initial_gyro_bias_cov": 0.01,
    "ekf_initial_orientation_cov": 0.1,
    "mag_ref_vector_ned": [22.0, 0.0, 45.0], # Example values for North, East, Down

    # EKF Tuning Settings
    "ekf_tune_samples": 300,        
    "min_ekf_r_accel_noise": 0.001, 
    "min_ekf_r_mag_noise": 0.01,    

    # Madgwick Tuning Settings
    "madgwick_tune_samples": 300,   
    "min_madgwick_beta": 0.005,     
    "max_madgwick_beta": 0.5,       
}

SETTINGS_FILE = "app_settings.json"

class SettingsManager:
    def __init__(self):
        self.settings = {}
        self.load_settings()

    def load_settings(self):
        logger.info(f"Attempting to load settings from {SETTINGS_FILE}")
        current_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    loaded_settings = json.load(f)
                current_settings = loaded_settings
                logger.info(f"Settings successfully loaded from {SETTINGS_FILE}")
            except Exception as e:
                logger.error(f"Error loading settings from {SETTINGS_FILE}: {e}. Using defaults for missing/invalid keys.", exc_info=True)

        # Start with defaults, then overlay loaded settings
        temp_settings = DEFAULT_SETTINGS.copy()
        temp_settings.update(current_settings) # Loaded settings override defaults
        self.settings = temp_settings

        # Ensure all default keys are present and potentially prune old keys
        # Also, check for deprecated keys like "use_ekf_filter"
        updated_config = False
        final_settings = {}
        
        if "use_ekf_filter" in self.settings: # Handle deprecated key
            logger.info("Deprecated setting 'use_ekf_filter' found. This is now controlled by 'ahrs_algorithm'. Removing.")
            del self.settings["use_ekf_filter"]
            updated_config = True

        for key, default_value in DEFAULT_SETTINGS.items():
            if key in self.settings:
                # Type check/coercion could be added here if necessary, but 'set' handles it.
                final_settings[key] = self.settings[key]
            else:
                logger.info(f"Adding missing setting '{key}' with default value '{default_value}'.")
                final_settings[key] = default_value
                updated_config = True
        
        # Check for keys in loaded settings that are no longer in defaults (they are stale)
        stale_keys = [key for key in self.settings if key not in DEFAULT_SETTINGS]
        if stale_keys:
            for stale_key in stale_keys:
                logger.info(f"Removing stale setting '{stale_key}' from configuration.")
                updated_config = True
        
        self.settings = final_settings # Use the cleaned and completed settings map

        if not os.path.exists(SETTINGS_FILE) or updated_config:
             logger.info(f"{SETTINGS_FILE} not found or was updated/migrated. Saving current configuration.")
             self.save_settings()


    def save_settings(self):
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logger.info(f"Settings saved to {SETTINGS_FILE}")
        except Exception as e:
            logger.error(f"Error saving settings to {SETTINGS_FILE}: {e}", exc_info=True)

    def get(self, key):
        # Fallback to DEFAULT_SETTINGS.get(key) ensures if key somehow got removed from self.settings but is in DEFAULT_SETTINGS
        return self.settings.get(key, DEFAULT_SETTINGS.get(key))

    def set(self, key, value_str):
        if key not in DEFAULT_SETTINGS:
            logger.warning(f"Attempted to set unknown or deprecated setting '{key}'. Ignoring.")
            return False

        default_val = DEFAULT_SETTINGS[key]
        new_value = None

        try:
            if isinstance(default_val, bool):
                if isinstance(value_str, bool): # Allow direct boolean input
                    new_value = value_str
                elif str(value_str).lower() in ['true', '1', 'yes', 'on']:
                    new_value = True
                elif str(value_str).lower() in ['false', '0', 'no', 'off']:
                    new_value = False
                else:
                    raise ValueError(f"Invalid boolean string: '{value_str}'")
            elif isinstance(default_val, float):
                new_value = float(value_str)
            elif isinstance(default_val, int):
                new_value = int(value_str)
            elif isinstance(default_val, list):
                # Try parsing as JSON list first, then as comma-separated string
                if isinstance(value_str, list): # Allow direct list input
                    parsed_list = value_str
                else: # Assume string input
                    try:
                        parsed_list = json.loads(str(value_str))
                    except json.JSONDecodeError:
                        if isinstance(value_str, str): # Must be a string to attempt split
                            parts = [p.strip() for p in value_str.replace('[','').replace(']','').split(',')]
                            if not all(parts): # Handle cases like "[,,]" or empty strings from split
                                 raise ValueError("List contains empty elements after parsing.")
                            parsed_list = [float(p) for p in parts] # Assume list of floats
                        else: # Not a list, not a string parseable to list
                            raise ValueError("Value is not a list and cannot be parsed from string.")


                if not isinstance(parsed_list, list):
                    raise ValueError("Parsed value is not a list.")
                # Example type check for list elements (assuming numeric if default is numeric)
                if default_val and all(isinstance(x, (int, float)) for x in default_val):
                    if not all(isinstance(x, (int, float)) for x in parsed_list):
                        raise ValueError("List elements must be numeric for this setting.")
                new_value = parsed_list
            elif key == "ahrs_algorithm": 
                if str(value_str) in ["EKF", "Madgwick"]:
                    new_value = str(value_str)
                else:
                    raise ValueError(f"Invalid AHRS algorithm: {value_str}. Must be 'EKF' or 'Madgwick'.")
            else: # Default to string
                new_value = str(value_str) 

            self.settings[key] = new_value
            logger.debug(f"Setting '{key}' updated to: {new_value} (type: {type(new_value)})")
            return True

        except ValueError as e:
            logger.warning(f"Could not convert setting '{key}' value '{value_str}' to expected type {type(default_val)}. Error: {e}. Keeping previous value: {self.settings.get(key)}")
            return False
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON string for setting '{key}' value '{value_str}'. Error: {e}. Keeping previous value: {self.settings.get(key)}")
            return False


    def reset_to_defaults(self):
        logger.info("Resetting all settings to their default values.")
        self.settings = DEFAULT_SETTINGS.copy()
        self.save_settings()
        logger.info("Settings reset to defaults and saved.")