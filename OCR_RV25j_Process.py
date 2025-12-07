#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV25J_Process OCR Pipeline and data processing
Author: Improved for modular maintainability and performance

Pipeline:
    *_table.jpg  ->  OCR (PP-Structure) or existing *_tblXX.md
                  ->  parse HTML/MD table
                  ->  CLEAN BLANK COLUMNS
                  ->  CALCULATE COORDINATES: Meter + Fraction/1000
                  ->  clean numeric
                  ->  detect closure
                  ->  *_OCR.toml  <-- CORRECTED OUTPUT NAME
"""

import argparse
import re
import sys
from pathlib import Path
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np

# NEW: Import the configuration manager
# NOTE: Assuming CONFIG_AppRV25J.py is accessible in the environment's path
# or the current working directory when this script runs.
try:
    from CONFIG_AppRV25J import Config_AppRV25J 
except ImportError:
    print("[FATAL] Required class Config_AppRV25J not found. Ensure CONFIG_AppRV25J.py is available.")
    sys.exit(1)


# ---- TOML reader (No longer strictly needed for config loading, but kept for safe_float if needed elsewhere) ----
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # older Python
    import tomli as tomllib  # type: ignore

# ---- Helper Function for robust float conversion ----
def safe_float(s):
    """Converts string to float, handles empty string, and cleans non-digit/dot chars."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    try:
        # Clean non-numeric characters first (except for '.')
        cleaned = re.sub(r"[^0-9.]", "", s)
        
        # Handle multiple dots (keeping the first one)
        if cleaned.count(".") > 1:
            first, *rest = cleaned.split(".")
            cleaned = first + "." + "".join(rest)
            
        return float(cleaned)
    except Exception:
        return np.nan
# -----------------------------------------------------


class RV25jProcessor:
    def __init__(self, app_file_dir: Path, root_folder: str, skip_ocr: bool = False):
        self.root = Path(root_folder)
        self.skip_ocr = skip_ocr
        self.pipeline = None # Will be lazily loaded
        
        # 1. Initialize Config Manager (loads Global config)
        self.config_manager = Config_AppRV25J(app_file_dir=app_file_dir)
        
        if not self.root.is_dir():
            raise ValueError(f"[ERROR] Folder not found: {self.root}")

        # 2. Load Local config from the root_folder (project-specific config.toml)
        # This will merge the local config onto the global one.
        print(f"[INFO] Loading local configuration from {self.root}...")
        try:
            self.config_manager.update_with_local(self.root)
            self.CONFIG = self.config_manager.CONFIG # Reference to the final merged Series
        except Exception as e:
            # Catch errors during local config loading, though update_with_local 
            # handles missing files gracefully, we catch parsing/critical errors.
            sys.exit(f"[FATAL] Failed to load or parse local config: {e}")

        # 3. Load COLUMN_SPEC from the merged configuration
        # NOTE: Based on provided TOMLs, the key is COLSPEC_RV25J, not [META].COLUMN_SPEC
        try:
            self.COLUMN_SPEC = self.CONFIG["COLSPEC_RV25J"]
        except KeyError:
            sys.exit("[FATAL] Configuration missing key: COLSPEC_RV25J")

        # Removed the heavy OCR initialization from __init__

    def _init_ocr_pipeline(self):
        """Initializes the heavy PaddleOCR model only when actually needed."""
        if not self.skip_ocr and self.pipeline is None:
            print("[INFO] Init PaddleOCR Thai PP-StructureV3...")
            # Import paddleocr here to ensure it's not imported unnecessarily
            try:
                from paddleocr import PPStructureV3
            except ImportError:
                print("[FATAL] paddleocr/PPStructureV3 not found. Cannot run OCR.")
                sys.exit(1)

            self.pipeline = PPStructureV3(
                lang="th",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_table_recognition=True,
            )

    # -----------------------------------------------------------
    def get_prefix(self, image_path: Path) -> str:
        stem = image_path.stem
        return stem[:-len("_table")] if stem.endswith("_table") else stem

    def _ColumnMeterFraction(self, df_raw) -> pd.DataFrame:
        # --- Step 3: Calculation of NORTHING/EASTING (Meters + Fraction/1000) ---
        
        # 3a. Rename columns for explicit calculation
        # The indices (5, 6, 7, 8) are based on the expected *original* 10-column structure.
        # This is a fixed assumption based on the data format.
        try:
            coord_map = {
                df_raw.columns[0]: self.COLUMN_SPEC[0], # Marker Name (e.g., 'MRK_DOL')
                df_raw.columns[5]: 'N_M',    # Column 6: Northing Meters
                df_raw.columns[6]: 'N_F',    # Column 7: Northing Fraction
                df_raw.columns[7]: 'E_M',    # Column 8: Easting Meters
                df_raw.columns[8]: 'E_F',    # Column 9: Easting Fraction
            }
            # Note: This operation may fail if the number of columns is not 9/10 after cleaning
            df_raw.rename(columns=coord_map, inplace=True)
        except IndexError:
            print("[WARN] Column index mismatch in _ColumnMeterFraction. Skipping coordinate calculation for this table.")
            return df_raw # Return original if indexing failed

        
        # 3b. Apply OCR correction (O->0, I->1, etc.) and safe float conversion
        coord_cols = ['N_M', 'N_F', 'E_M', 'E_F']
        
        # Correct marker column first (based on the rename map)
        marker_col = self.COLUMN_SPEC[0]
        if marker_col in df_raw.columns:
            df_raw[marker_col] = (df_raw[marker_col].astype(str).str.replace('O', '0')
                                 .str.replace('o', '0').str.replace('I', '1')
                                 .str.replace('i', '1').str.replace('l', '1')
                                 .str.replace('L', '1').str.strip())
        
        for col in coord_cols:
            if col in df_raw.columns:
                # Apply safe_float (includes internal cleaning) for numeric data
                df_raw[col] = df_raw[col].apply(safe_float)
            else:
                 # If a column is missing (e.g. dropped blank), fill with NaN to prevent crash
                 df_raw[col] = np.nan 
                 
        # 3c. Calculate final coordinates
        df_raw[self.COLUMN_SPEC[1]] = df_raw['N_M'] + (df_raw['N_F'] / 1000.0)
        df_raw[self.COLUMN_SPEC[2]] = df_raw['E_M'] + (df_raw['E_F'] / 1000.0)
        return df_raw

    # -----------------------------------------------------------
    def parse_markdown_table(self, md_path: Path) -> pd.DataFrame:
        MRK_COL,N_COL,E_COL = self.COLUMN_SPEC 
        html = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        if not table:
            print(f"[WARN] No <table> in {md_path}")
            return pd.DataFrame(columns=[c for c in self.COLUMN_SPEC if c])

        try:
            # Use pandas to parse the HTML table back into a DataFrame
            df_raw = pd.read_html(StringIO(str(table)))[0].reset_index(drop=True)
        except Exception as e:
            print(f"[WARN] pandas.read_html failed {md_path}: {e}")
            return pd.DataFrame(columns=[c for c in self.COLUMN_SPEC if c])

        # Step 1: Clean up strings (strip whitespace, replace non-breaking space)
        df_raw = df_raw.map(
            lambda x: "" if pd.isna(x) else str(x).replace("\xa0", " ").strip()
        )

        # Step 2: Identify and drop columns that are entirely blank
        cols_to_drop = [
            col for col in df_raw.columns
            if all(val == "" for val in df_raw[col])
        ]

        if cols_to_drop:
            print(f"[INFO] Dropping blank columns: {cols_to_drop}")
            df_raw = df_raw.drop(columns=cols_to_drop)

        # Determine calculation path based on column count after cleaning
        if len(df_raw.columns) >= 9:
            # This is the expected full coordinate table (Marker + 4 pairs)
            df_raw = self._ColumnMeterFraction(df_raw)
        elif len(df_raw.columns) >= 3:
            # Assume it's a simplified table where final Northing/Easting are already present 
            # (e.g., columns 0, -2, -1)
            try:
                coord_map = {
                    df_raw.columns[ 0]: MRK_COL, 
                    df_raw.columns[-2]: N_COL, 
                    df_raw.columns[-1]: E_COL, 
                    }
                df_raw.rename(columns=coord_map, inplace=True)
            except IndexError:
                 print(f"[WARN] Cannot map columns for simplified table in {md_path}.")
                 return pd.DataFrame(columns=[c for c in self.COLUMN_SPEC if c])
        else:
            print(f"[WARN] Insufficient columns ({len(df_raw.columns)}) to process table in {md_path}.")
            return pd.DataFrame(columns=[c for c in self.COLUMN_SPEC if c])


        # --- Step 4: Final Cleaning and Filtering ---

        # 4a. Explicitly ensure calculated coordinates are clean floats 
        for col in [N_COL, E_COL]:
            # This handles both cases: calculated floats, or raw strings from simplified table
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        # 4b. Filter out rows where both calculated coordinates are NaN
        df_raw.dropna(subset=[N_COL, E_COL], how='all', inplace=True)

        # 4c. Remove all extra columns, keeping only the final required columns
        out_cols = [c for c in self.COLUMN_SPEC if c] 
        # Select ONLY the final required columns
        df_final = df_raw[out_cols].copy() 

        # --- Step 5: Final Formatting for TOML output ---
        rows = []
        marker_col = self.COLUMN_SPEC[0]
        
        for _, rec in df_final.iterrows():
            rec_dict = rec.to_dict() 
            
            # Format coordinates to 3 decimal places
            try:
                rec_dict[N_COL] = f"{rec_dict[N_COL]:.3f}"
            except Exception:
                rec_dict[N_COL] = ""
            
            try:
                rec_dict[E_COL] = f"{rec_dict[E_COL]:.3f}"
            except Exception:
                rec_dict[E_COL] = ""
            
            # Final check and stripping for marker
            rec_dict[MRK_COL] = str(rec_dict[MRK_COL]).strip()

            # Filter: Only append rows that have at least one valid coordinate value
            if rec_dict[N_COL] or rec_dict[E_COL]:
                rows.append(rec_dict)

        return pd.DataFrame(rows, columns=out_cols)

    # -----------------------------------------------------------
    # ... (find_images, filter_images, list_files methods unchanged) ...
    def find_images(self) -> list[Path]:
        """Helper to return sorted list of matching images."""
        return sorted(self.root.rglob("*_table.jpg"))

    def filter_images(self, images: list[Path], range_str: str) -> list[Path]:
        """
        Filters the list of images based on a 1-based range string 'start,end'.
        Example: "4,6" -> returns images at index 3, 4, 5 (User's 4, 5, 6)
        """
        if not range_str:
            return images

        total = len(images)
        try:
            if "," in range_str:
                parts = range_str.split(",")
                if len(parts) != 2:
                    raise ValueError
                start_idx = int(parts[0].strip())
                end_idx = int(parts[1].strip())
            else:
                # Single number case
                start_idx = int(range_str.strip())
                end_idx = start_idx

            # Bounds checking (Clamp values)
            if start_idx < 1: start_idx = 1
            if end_idx > total: end_idx = total
            
            if start_idx > end_idx:
                print(f"[WARN] Invalid range {start_idx}-{end_idx}. Processing nothing.")
                return []

            # Convert 1-based user input to 0-based Python slice
            # Slice is [start-1 : end]
            subset = images[start_idx-1 : end_idx]
            
            print(f"[INFO] Image Range: {start_idx} to {end_idx} (Selected {len(subset)} files)")
            return subset

        except ValueError:
            print(f"[ERROR] Invalid format for -i/--images: '{range_str}'. Expected 'start,end' (e.g. '4,6')")
            sys.exit(1)

    def list_files(self):
        """Lists all matching files with Index IDs and exits."""
        images = self.find_images()
        if not images:
            print(f"[INFO] No *_table.jpg found in: {self.root}")
            return

        print(f"[INFO] Found {len(images)} files in: {self.root}")
        print("-" * 60)
        for idx, img in enumerate(images, start=1):
            try:
                display_path = img.relative_to(self.root)
            except ValueError:
                display_path = img
            # Display index number [N] to help user select range
            print(f"[{idx}] {display_path}")
        print("-" * 60)

    # -----------------------------------------------------------
    def run_ocr(self, image_path: Path) -> pd.DataFrame:
        self._init_ocr_pipeline() # Initialize the model here
        if self.pipeline is None:
             print("[ERROR] OCR pipeline failed to initialize.")
             return pd.DataFrame(columns=self.COLUMN_SPEC)

        prefix = self.get_prefix(image_path)
        out_img_dir = image_path.parent / "imgs"
        out_img_dir.mkdir(exist_ok=True)

        print(f"\n[INFO] OCR: {image_path}")
        outputs = self.pipeline.predict(str(image_path))

        dfs = []
        for i, res in enumerate(outputs):
            md_file = image_path.parent / f"{prefix}_tbl{i:02d}.md"
            # NOTE: We save the MD output *before* parsing, so -s can use it later
            res.save_to_markdown(save_path=str(md_file)) 
            res.save_to_img(save_path=str(out_img_dir))

            df = self.parse_markdown_table(md_file)
            if not df.empty:
                dfs.append(df)

        return (
            pd.concat(dfs, ignore_index=True)
            if dfs
            else pd.DataFrame(columns=self.COLUMN_SPEC)
        )

    # -----------------------------------------------------------
    def parse_existing_md(self, image_path: Path) -> pd.DataFrame:
        prefix = self.get_prefix(image_path)
        md_files = sorted(image_path.parent.glob(f"{prefix}_tbl*.md"))

        if not md_files:
            print(f"[WARN] No MD found: {image_path}. Try running without -s.")
            return pd.DataFrame(columns=[c for c in self.COLUMN_SPEC if c])

        dfs = [self.parse_markdown_table(md) for md in md_files]
        dfs = [df for df in dfs if not df.empty]
        return (
            pd.concat(dfs, ignore_index=True)
            if dfs
            else pd.DataFrame(columns=self.COLUMN_SPEC)
        )

    # -----------------------------------------------------------
    def _toml_escape(self, s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # -----------------------------------------------------------
    def get_meta_and_deed_from_config(self):
        # Accessing merged configuration using flattened keys
        
        # Access DOL_Office (flat key)
        office = self.CONFIG.get("DOL_Office")
        if office is None:
            sys.exit("[FATAL] Configuration missing key: DOL_Office")

        if not isinstance(office, str) or not office.strip():
            sys.exit(f"[FATAL] Invalid DOL_Office: {office}")

        # --- REFACTORED: Use the correct flat keys from TOML (SURVEY_TYPE and DEED_EPSG) ---
        survey_type = self.CONFIG.get("SURVEY_TYPE") 
        epsg = self.CONFIG.get("DEED_EPSG")         
        
        if survey_type is None or epsg is None:
            # Corrected error message to match actual keys
            sys.exit("[FATAL] Configuration missing key: SURVEY_TYPE or DEED_EPSG") 


        if not isinstance(survey_type, str) or not survey_type.strip():
            sys.exit(f"[FATAL] Invalid SURVEY_TYPE: {survey_type}")

        if isinstance(epsg, int):
            epsg_str = str(epsg)
        elif isinstance(epsg, str) and epsg.strip().isdigit():
            epsg_str = epsg.strip()
        else:
            sys.exit(f"[FATAL] Invalid DEED_EPSG: {epsg}")

        return office, survey_type, epsg_str

    # -----------------------------------------------------------
    def write_toml(self, image_path: Path, df: pd.DataFrame):
        prefix = self.get_prefix(image_path)
        toml_path = image_path.with_name(f"{prefix}_OCR.toml")

        vertices = []
        MRK_COL,N_COL,E_COL = self.COLUMN_SPEC

        # Note: df now only contains the three required columns (e.g., Marker, Northing, Easting)
        for _, r in df.iterrows():
            try:
                # The values are strings in '0.000' format
                n = float(r[N_COL])
                e = float(r[E_COL])
                vertices.append({"marker": r[MRK_COL], "north": n, "east": e})
            except Exception:
                continue

        if not vertices:
            print(f"[WARN] No numeric rows found for TOML output: {image_path}")
            return []
       
        rows = []
        for idx, v in enumerate(vertices, start=1):
            label = chr(64 + idx) if idx <= 26 else f"P{idx}"
            rows.append([idx, label, v["marker"], v["north"], v["east"]])

        office, survey_type, epsg_str = self.get_meta_and_deed_from_config()

        lines = []
        lines.append("[META]")
        lines.append(f'DOL_Office = "{self._toml_escape(office)}"')
        lines.append("")
        lines.append("[Deed]")
        lines.append('ParcelNumber = "000"')
        lines.append('MapSheet = "DDDD-II-DDDD"')
        # Output TOML uses standard key names for Deed section
        lines.append(f'Survey_Type = "{self._toml_escape(survey_type)}"')
        lines.append(f"EPSG = {epsg_str}")
        lines.append('unit = "meter"')
        lines.append(f'area_grid = "rai-ngan-wa"')
        lines.append(f'area_topo = "rai-ngan-wa"')
        lines.append("marker = [")

        for idx, label, name, n, e in rows:
            lines.append(
                f'  [{idx}, "{self._toml_escape(label)}", '
                f'"{self._toml_escape(name)}", {n:.3f}, {e:.3f}],'
            )
        lines.append("]")

        toml_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] TOML -> {toml_path}")
        return vertices

    # -----------------------------------------------------------
    def process(self, image_range: str = None):
        all_images = self.find_images()
        if not all_images:
            sys.exit("[ERROR] No *_table.jpg found")

        # FILTER IMAGES IF RANGE IS PROVIDED
        images_to_process = self.filter_images(all_images, image_range)

        if not images_to_process:
            print("[INFO] No images to process based on filter.")
            return

        print(f"[INFO] Processing {len(images_to_process)} / {len(all_images)} detected files.")

        for img in images_to_process:
            print("\n" + "=" * 70)
            print(f"[PROCESS] {img}")

            # Uses parse_existing_md if -s is true, otherwise uses run_ocr
            df = self.parse_existing_md(img) if self.skip_ocr else self.run_ocr(img)

            if df.empty:
                print("[WARN] Empty DF from OCR/MD")
                continue
            
            self.write_toml(img, df)

        print("\n[DONE] Processing complete.")


# ============================================================
# CLI Entry
# ============================================================
def main():
    # Expanded description for usage and examples
    epilog_text = """
Usage Examples for viewing results (assuming common Linux CLI tools are installed):

1) View source images (*_table.jpg):
   find . -type f -iname "*_table.jpg" -print0 | xargs -0 -n1 eog

2) View raw OCR markdown results (*_tbl00.md):
   find . -type f -iname "*_tbl00.md" -print0 | xargs -0 -I {} sh -c 'w3m -dump -T text/html "{}"'

3) View final Deed TOML file (*_OCR.toml):
   find . -type f -name "*_OCR.toml" -print0 | xargs -0 batcat
"""
    parser = argparse.ArgumentParser(
        description="RV25j OCR -> TOML processing script.",
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("folder", help="Folder containing *_table.jpg")
    parser.add_argument(
        "-s", "--skip-ocr",
        action="store_true",
        help="Skip the OCR process and use existing *_tbl*.md files for parsing."
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all matching files with ID numbers and exit."
    )
    # ADDED: Images range argument
    parser.add_argument(
        "-i", "--images",
        type=str,
        help="Range of image numbers to process (e.g., '4,6' for 4, 5, 6). Use -l to see numbers."
    )
    
    args = parser.parse_args()
    
    # Determine the directory where this script resides for Global config loading
    app_file_dir = Path(sys.argv[0]).resolve().parent
    
    # Initialize the processor with basic configuration loaded.
    try:
        # Pass app_file_dir (for global config) and args.folder (for local config)
        processor = RV25jProcessor(app_file_dir=app_file_dir, root_folder=args.folder, skip_ocr=True)
    except SystemExit as e:
        # Catch and re-raise SystemExit from __init__ for config errors
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Initialization error: {e}")
        sys.exit(1)
    
    if args.list:
        processor.list_files()
        # Exit immediately after listing, before any heavy OCR initialization
        return
    
    # If we are processing, set the correct skip_ocr flag for the processor
    # and proceed. The OCR model will only be loaded inside run_ocr() if skip_ocr is False.
    processor.skip_ocr = args.skip_ocr
    processor.process(image_range=args.images)


if __name__ == "__main__":
    main()