import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any

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
    
    The configuration is stored internally as flat Series (e.g., 'Deed.EPSG' key).
    Public properties (GLOBAL_CONFIG, LOCAL_CONFIG, CONFIG) expose the Series 
    directly for access via dictionary keys (e.g., config.CONFIG['Deed.EPSG']).
    """
    
    def __init__(self, app_file_dir: Path, global_filename: str = "CONFIG_AppRV25J.toml", local_filename: str = "config.toml"):
        """
        Initializes configuration file names and loads the GLOBAL configuration immediately.
        
        Args:
            app_file_dir: The directory where the main application file resides.
            global_filename: Name of the global configuration file.
            local_filename: Name of the local configuration file.
        """
        # Store filenames as instance attributes
        self.global_filename = global_filename
        self.local_filename = local_filename
        
        # Internal Series storage
        self._GLOBAL_CONFIG_series: pd.Series = pd.Series(dtype=object)
        self._LOCAL_CONFIG_series: pd.Series = pd.Series(dtype=object)
        self._CONFIG_series: pd.Series = pd.Series(dtype=object) # Final merged config
        
        # REQUIREMENT: Load GLOBAL config immediately upon initialization
        self.load_initial_global(app_file_dir)
        
    # --- Public Accessor Properties (Return Series directly) ---
    
    @property
    def GLOBAL_CONFIG(self) -> pd.Series:
        """Accessor for Global configuration (read-only, flat keys)."""
        return self._GLOBAL_CONFIG_series
        
    @property
    def LOCAL_CONFIG(self) -> pd.Series:
        """Accessor for Local configuration (read-only, flat keys)."""
        return self._LOCAL_CONFIG_series
        
    @property
    def CONFIG(self) -> pd.Series:
        """Accessor for the Final merged configuration (read-only, flat keys)."""
        return self._CONFIG_series

    # --- Internal Methods ---

    def _flatten_toml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively flattens a nested TOML dictionary structure into a single 
        level key-value dictionary using dots for hierarchy (e.g., 'Deed.EPSG').
        """
        flattened = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # Recursively flatten nested dictionaries
                nested_flat = self._flatten_toml(v)
                for nk, nv in nested_flat.items():
                    flattened[f"{k}.{nk}"] = nv
            else:
                flattened[k] = v
        return flattened

    def _read_toml_to_df(self, file_path: Path) -> pd.Series:
        """Reads a TOML file, flattens its content, and returns a single Series."""
        if not file_path.is_file():
            raise FileNotFoundError(f"TOML file not found at: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)
            
        # Flatten the dictionary to handle any sections (like [Deed])
        flat_data = self._flatten_toml(data)
        
        # Convert to a Series 
        df = pd.Series(flat_data, dtype=object)
        return df

    def load_initial_global(self, app_file_dir: Path):
        """
        Loads the GLOBAL configuration and initializes self.CONFIG with it.
        Called only once from __init__.
        """
        global_path = app_file_dir / self.global_filename
        
        print("-" * 50)
        print("Starting Cascading Configuration Load (Global)...")
        
        try:
            self._GLOBAL_CONFIG_series = self._read_toml_to_df(global_path)
            # Initialize final config with global settings
            self._CONFIG_series = self._GLOBAL_CONFIG_series.copy() 
            print(f"SUCCESS: Global configuration loaded from {self.global_filename}")
        except FileNotFoundError as e:
            print(f"WARNING: Global config not found. {e}")
        except Exception as e:
            print(f"ERROR: Failed to load Global config: {e}")

    def _load_local_config(self, working_folder: Path):
        """
        Loads the project-specific configuration (config.toml).
        Internal use for update_with_local.
        """
        local_path = working_folder / self.local_filename
        
        try:
            self._LOCAL_CONFIG_series = self._read_toml_to_df(local_path)
            print(f"SUCCESS: Local configuration loaded from {self.local_filename}")
        except FileNotFoundError as e:
            # Clear local series if file not found to prevent merging empty data
            self._LOCAL_CONFIG_series = pd.Series(dtype=object)
            print(f"WARNING: Local config not found. {e}")
        except Exception as e:
            # Clear local series on error
            self._LOCAL_CONFIG_series = pd.Series(dtype=object)
            print(f"ERROR: Failed to load Local config: {e}")

    def update_with_local(self, working_folder: Path):
        """
        Loads the local config and merges it onto the existing 
        self._CONFIG_series (Local overwrites Global).
        
        Performs a version check against the 'VERSION' key before merging.
        """
        self._load_local_config(working_folder)
        
        if self._LOCAL_CONFIG_series.empty:
            print("INFO: No valid local configuration to merge.")
            print("-" * 50)
            return

        # 1. Get versions for comparison
        global_version = self._GLOBAL_CONFIG_series.get('VERSION')
        local_version = self._LOCAL_CONFIG_series.get('VERSION')
        
        # 2. Perform version check (must match exactly)
        if global_version is None or local_version is None or global_version != local_version:
            print("ERROR: VERSION MISMATCH OR MISSING VERSION KEY.")
            print(f"       Global VERSION: {global_version}")
            print(f"       Local VERSION:  {local_version}")
            print("       Local configuration will be IGNORED.")
            # Do NOT merge local config
            self._LOCAL_CONFIG_series = pd.Series(dtype=object) # Clear it for safety
            
        else:
            # 3. Merge if versions match
            # Local Config (priority) combines with Global Config.
            self._CONFIG_series = self._LOCAL_CONFIG_series.combine_first(self._CONFIG_series)
            print(f"SUCCESS: Configurations merged (Version {global_version} verified).")
        
        print("-" * 50)


# =================================================================
# --- Demonstration of Usage ---
# =================================================================

def demonstration():
    """
    Demonstrates how to use the Config_AppRV25J class with split loading 
    and verifies the data types of the final configuration, using standard Series access.
    """
    print("\n" + "="*50)
    print("DEMONSTRATION: Config_AppRV25J (Direct Series Access)")
    print("="*50)
    
    # --- Mocking File Paths for Demonstration ---
    MOCK_APP_DIR = Path.cwd()
    MOCK_WORKING_FOLDER = Path.cwd() 
    
    # --- Step 1: Instantiate and Load GLOBAL config in __init__ ---
    print("\n--- Step 1: Load GLOBAL Config (via __init__) ---")
    config_manager = Config_AppRV25J(app_file_dir=MOCK_APP_DIR)
    
    # --- Step 2: Load LOCAL config and Merge (Includes Version Check) ---
    print("\n--- Step 2: Load LOCAL Config and Merge ---")
    config_manager.update_with_local(MOCK_WORKING_FOLDER)

    # --- Step 3: Verify and Print Types ---
    
    print("\n--- Final CONFIG Data & Types (Merged) ---")
    
    final_config = config_manager.CONFIG # Access the Series via the property
    
    if not final_config.empty:
        
        print(f"{'Key':<30} | {'Value':<18} | {'Type'}")
        print("-" * 65)

        # Iterate through the final configuration Series
        for key, value in final_config.items():
            display_value = str(value)[:18]
            value_type = type(value).__name__
            if 'numpy' in value_type.lower() or 'float' in value_type.lower():
                 value_type = 'float'
            
            print(f"{key:<30} | {display_value:<18} | {value_type}")
            
        print("\n--- Verification using Standard Series Access (Flat Keys) ---")
        
        # Verification using standard Series access with expected FLAT keys
        try:
            # Keys expected from GLOBAL:
            print(f"config.CONFIG['COLSPEC_TOML']: {final_config.get('COLSPEC_TOML', 'N/A')}")
            
            # Keys expected from GLOBAL/LOCAL (Flat keys):
            print(f"config.CONFIG['DOL_Office']: {final_config.get('DOL_Office', 'N/A')}")
            print(f"config.CONFIG['SURVEY_TYPE']: {final_config.get('SURVEY_TYPE', 'N/A')}")
            print(f"config.CONFIG['DEED_EPSG']: {final_config.get('DEED_EPSG', 'N/A')}")
            print(f"config.CONFIG['VERSION']: {final_config.get('VERSION', 'N/A')}")
            
        except KeyError as e:
            # If this hits, the key structure is inconsistent.
            print(f"Key Error: {e} not found in configuration. Check TOML files.")
            
    else:
        print("Final CONFIG is empty.")

if __name__ == '__main__':
    # NOTE: You must ensure 'CONFIG_AppRV25J.toml' and 'config.toml' 
    # are present in the current working directory, and the VERSION keys match, 
    # for the local config to be loaded successfully in the demonstration.
    demonstration()