import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List # Added List for return type

# --- TOML Library Import ---
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        # If tomli is also missing, exit gracefully
        print("Error: Missing 'tomllib' or 'tomli'. Please install the 'tomli' package.", file=sys.stderr)
        sys.exit(1)

# =================================================================
# --- Configuration Manager Class ---
# =================================================================

class Config_AppRV25J:
    """
    Manages application configuration using a cascading load sequence from two
    TOML files, storing the data in Pandas Series.
    """
    
    def __init__(self, app_file_dir: Path, global_filename: str = "CONFIG_AppRV25J.toml", local_filename: str = "config.toml"):
        """
        Initializes configuration file names and internal Series storage.
        
        NOTE: GLOBAL configuration is NO LONGER loaded here to allow the caller 
        (AppRV25J_Center) to capture log messages from load_global_config_and_log.
        """
        self.app_file_dir = app_file_dir
        self.global_filename = global_filename
        self.local_filename = local_filename
        
        self._GLOBAL_CONFIG_series: pd.Series = pd.Series(dtype=object)
        self._LOCAL_CONFIG_series: pd.Series = pd.Series(dtype=object)
        self._CONFIG_series: pd.Series = pd.Series(dtype=object)
        
    # --- Public Accessor Properties (Return Series directly) ---
    @property
    def GLOBAL_CONFIG(self) -> pd.Series:
        return self._GLOBAL_CONFIG_series
        
    @property
    def LOCAL_CONFIG(self) -> pd.Series:
        return self._LOCAL_CONFIG_series
        
    @property
    def CONFIG(self) -> pd.Series:
        return self._CONFIG_series

    # --- Internal Methods ---

    def _flatten_toml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively flattens a nested TOML dictionary structure."""
        flattened = {}
        for k, v in data.items():
            if isinstance(v, dict):
                nested_flat = self._flatten_toml(v)
                for nk, nv in nested_flat.items():
                    flattened[f"{k}.{nk}"] = nv
            else:
                flattened[k] = v
        return flattened

    def _read_toml_to_df(self, file_path: Path) -> pd.Series:
        """Reads a TOML file, flattens its content, and returns a single Series."""
        if not file_path.is_file():
            # Raise FileNotFoundError which is handled by the caller
            raise FileNotFoundError(f"TOML file not found at: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)
            
        flat_data = self._flatten_toml(data)
        df = pd.Series(flat_data, dtype=object)
        return df

    def load_global_config_and_log(self) -> List[str]:
        """
        Loads the GLOBAL configuration and initializes self.CONFIG with it.
        Returns: List of log messages.
        """
        log_messages = []
        global_path = self.app_file_dir / self.global_filename
        
        log_messages.append("-" * 50)
        log_messages.append("Starting Cascading Configuration Load (Global)...")
        
        try:
            self._GLOBAL_CONFIG_series = self._read_toml_to_df(global_path)
            self._CONFIG_series = self._GLOBAL_CONFIG_series.copy() 
            log_messages.append(f"SUCCESS: Global configuration loaded from {self.global_filename}")
            
            # --- New Logic to Check and Log VERSION ---
            # Get the VERSION key, defaulting to 'UNKNOWN' if missing
            version = self._GLOBAL_CONFIG_series.get('VERSION', 'UNKNOWN')
            log_messages.append(f"INFO: Configuration Manager Version: {version}")
            # --- End New Logic ---
            
        except FileNotFoundError as e:
            log_messages.append(f"WARNING: Global config not found. {e}")
        except Exception as e:
            # Re-raise exceptions here, as configuration failure is FATAL for the application
            log_messages.append(f"ERROR: Failed to load Global config: {e}")
            raise RuntimeError(f"Global configuration failed to load.") from e
            
        return log_messages

    def _load_local_config(self, working_folder: Path) -> List[str]:
        """
        Loads the project-specific configuration (config.toml).
        Returns: List of log messages.
        """
        log_messages = []
        local_path = working_folder / self.local_filename
        
        try:
            self._LOCAL_CONFIG_series = self._read_toml_to_df(local_path)
            log_messages.append(f"SUCCESS: Local configuration loaded from {self.local_filename}")
        except FileNotFoundError as e:
            log_messages.append(f"WARNING: Local config not found at {local_path.name}.")
        except Exception as e:
            log_messages.append(f"ERROR: Failed to load Local config: {e}")
            
        return log_messages

    def update_with_local(self, working_folder: Path) -> List[str]:
        """
        Loads the local config and merges it onto the existing 
        self._CONFIG_series (Local overwrites Global).
        Returns: List of log messages.
        """
        log_messages = self._load_local_config(working_folder)
        
        if self._CONFIG_series.empty and self._LOCAL_CONFIG_series.empty:
            log_messages.append("WARNING: Cannot merge, both configurations are empty.")
            log_messages.append("-" * 50)
            return log_messages

        if not self._LOCAL_CONFIG_series.empty:
            
            # Use combine_first to ensure new keys are added and overlapping keys are overwritten.
            self._CONFIG_series = self._LOCAL_CONFIG_series.combine_first(self._CONFIG_series)
            log_messages.append("SUCCESS: Configurations merged (Local overwrote Global, new keys added).")
        
        log_messages.append("-" * 50)
        return log_messages

# If running independently for demonstration purposes (keep this part standard)
if __name__ == '__main__':
    print("This file contains the configuration manager class, not the main application.")