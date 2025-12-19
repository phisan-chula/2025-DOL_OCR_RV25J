#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV25j_Cadastre.py — RV25J Marker Processor (OOP, CONFIG.toml required)

Assumptions
-----------
1) Configuration is loaded using Config_AppRV25J, starting with CONFIG.toml 
   in the current working directory.
2) Marker source files are *_OCRedit.toml, containing the [Deed].marker array.
   Interpreted as: [SEQ_NUM, MRK_SEQ, MRK_DOL, NORTHING, EASTING]
3) Workflow includes: load markers, transform coordinates, write GPKG, 
   list accessed files, and archive source files (py7zr).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from pyproj import CRS, Transformer

# --- Import py7zr for archiving ---
try:
    import py7zr
except ImportError:
    print("Error: Missing 'py7zr'. Please run 'pip install py7zr'.", file=sys.stderr)
    sys.exit(1)

# --- Import Shared Configuration Manager ---
# Set up path to import Config_AppRV25J
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

try:
    from CONFIG_AppRV25J import Config_AppRV25J
except ImportError:
    print("Error: Could not import Config_AppRV25J. Check path and filename.", file=sys.stderr)
    sys.exit(1)

# --- TOML loader (kept for MarkerLoader to read *_OCRedit.toml) ---
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # fallback for older versions


# =========================================
# Utility Functions for Config Access
# =========================================

def _get_toml_spec(config_series: pd.Series) -> List[str]:
    """Retrieves TOML_SPEC, falling back to a default if not found."""
    default_spec = ["SEQ_NUM", "MRK_SEQ", "MRK_DOL", "NORTHING", "EASTING"]
    
    return (
        config_series.get("META.TOML_SPEC") or 
        config_series.get("TOML_SPEC") or 
        default_spec
    )

def _get_default_epsg(config_series: pd.Series) -> int:
    """Determine the default EPSG from the configuration."""
    epsg_val = config_series.get("DEED_EPSG") 
    if epsg_val is None:
        epsg_val = config_series.get("Deed.EPSG") or config_series.get("Deed.epsg")

    if epsg_val is not None:
        try:
            return int(epsg_val)
        except ValueError:
            pass 

    return 24047

def _get_towgs84(config_series: pd.Series) -> List[float] | None:
    """Retrieves towgs84 list."""
    towgs84 = config_series.get("META.towgs84") or config_series.get("towgs84")
    
    if isinstance(towgs84, list) and len(towgs84) >= 3:
        try:
            return [float(v) for v in towgs84]
        except ValueError:
            print(f"[WARNING] towgs84 found but contains non-numeric values: {towgs84!r}")
            return None
    
    if towgs84 is not None:
         print(f"[WARNING] towgs84 found but is not a list of 3+ elements: {towgs84!r}")
         
    return None

# =========================================
# CRS / Transformer factory 
# =========================================

class CRSFactory:
    """
    Build CRS for Indian 1975 UTM (EPSG 24047 / 24048) with towgs84 if provided.
    """

    def __init__(self, towgs84: List[float] | None):
        self.towgs84 = towgs84
        self._crs_cache: Dict[int, CRS] = {}
        self._transformer_cache: Dict[int, Transformer] = {}
        self._crs_wgs84 = CRS.from_epsg(4326)

        self._crs_w84_utm_cache: Dict[int, CRS] = {}
        self._transformer_w84_utm_cache: Dict[int, Transformer] = {}

    def _build_proj4_id75(self, epsg: int) -> CRS:
        if epsg == 24047:
            zone = 47
        elif epsg == 24048:
            zone = 48
        else:
            return CRS.from_epsg(epsg)

        towgs_str = ""
        if self.towgs84 and len(self.towgs84) >= 3:
            towgs_str = "+towgs84=" + ",".join(str(v) for v in self.towgs84) + " "

        proj4 = (
            f"+proj=utm +zone={zone} "
            f"+a=6377276.345 +rf=300.8017 "
            f"{towgs_str}"
            f"+units=m +no_defs"
        )
        return CRS.from_proj4(proj4)

    def get_src_crs(self, epsg: int) -> CRS:
        if epsg not in self._crs_cache:
            self._crs_cache[epsg] = self._build_proj4_id75(epsg)
        return self._crs_cache[epsg]

    def get_transformer_to_wgs84(self, epsg: int) -> Transformer:
        if epsg not in self._transformer_cache:
            crs_src = self.get_src_crs(epsg)
            self._transformer_cache[epsg] = Transformer.from_crs(
                crs_src, self._crs_wgs84, always_xy=True
            )
        return self._transformer_cache[epsg]

    def get_w84_utm_crs(self, epsg_src: int) -> CRS:
        if epsg_src in self._crs_w84_utm_cache:
            return self._crs_w84_utm_cache[epsg_src]

        if epsg_src in (24047, 32647):
            epsg_dst = 32647
        elif epsg_src in (24048, 32648):
            epsg_dst = 32648
        else:
            epsg_dst = epsg_src

        crs = CRS.from_epsg(epsg_dst)
        self._crs_w84_utm_cache[epsg_src] = crs
        return crs

    def get_transformer_to_w84_utm(self, epsg_src: int) -> Transformer:
        if epsg_src not in self._transformer_w84_utm_cache:
            crs_src = self.get_src_crs(epsg_src)
            crs_dst = self.get_w84_utm_crs(epsg_src)
            self._transformer_w84_utm_cache[epsg_src] = Transformer.from_crs(
                crs_src, crs_dst, always_xy=True
            )
        return self._transformer_w84_utm_cache[epsg_src]

    @property
    def crs_wgs84(self) -> CRS:
        return self._crs_wgs84

# =========================================
# MarkerLoader: read markers recursively 
# =========================================

class MarkerLoader:
    """
    - Recursively find *_OCRedit.toml under the given folder.
    - Read [Deed].marker.
    """

    def __init__(self, folder: Path, toml_spec: List[str], default_epsg: int):
        self.folder = folder
        self.toml_spec = toml_spec
        self.default_epsg = default_epsg
        self.found_toml_files: List[Path] = [] 

    @staticmethod
    def _file_prefix_from_path(path: Path) -> str:
        stem = path.stem
        suffix = "_OCRedit"
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
        return stem

    @staticmethod
    def _extract_epsg_from_toml(toml_data: dict, default_epsg: int) -> int:
        """Look for EPSG or crs inside [Deed] section."""
        deed = toml_data.get("Deed") or toml_data.get("deed")
        if not isinstance(deed, dict):
            return default_epsg

        epsg = deed.get("EPSG") or deed.get("epsg")
        if epsg is None:
            crs_val = deed.get("crs") or deed.get("CRS")
            if crs_val is not None:
                try:
                    epsg = int(crs_val)
                except ValueError:
                    epsg = None
        if epsg is None:
            return default_epsg
        return int(epsg)

    @staticmethod
    def _extract_markers_from_deed(toml_data: dict, toml_spec: List[str]):
        """
        Extracts markers assuming the structure:
        [idx, MRK_SEQ, MRK_DOL, NORTHING, EASTING]
        """
        rows = []
        deed = toml_data.get("Deed") or toml_data.get("deed")
        if not isinstance(deed, dict):
            return rows

        marker_arr = deed.get("marker")
        if not isinstance(marker_arr, list):
            return rows

        if len(toml_spec) < 5:
            print(f"[ERROR] TOML_SPEC has less than 5 elements. Cannot map marker array.")
            return rows

        for entry in marker_arr:
            if not isinstance(entry, (list, tuple)) or len(entry) < 5:
                continue

            idx_raw, marker_raw, code_raw, n_raw, e_raw = entry[:5]

            try:
                n_val = float(n_raw)
                e_val = float(e_raw)
            except Exception:
                continue

            rows.append(
                {
                    toml_spec[0]: idx_raw,
                    toml_spec[1]: marker_raw,
                    toml_spec[2]: code_raw,
                    toml_spec[3]: n_val,
                    toml_spec[4]: e_val,
                }
            )

        return rows

    def load_df_id75(self) -> pd.DataFrame:
        """
        Searches for *_OCRedit.toml and returns the combined DataFrame.
        """
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Folder not found: {self.folder}")

        toml_files = list(self.folder.rglob("*_OCRedit.toml"))
        self.found_toml_files = toml_files

        if not toml_files:
            raise FileNotFoundError(
                f"No *_OCRedit.toml files found under {self.folder}"
            )

        all_rows = []
        toml_spec = self.toml_spec
        
        for chosen in toml_files:
            file_prefix = self._file_prefix_from_path(chosen)

            try:
                with chosen.open("rb") as fp:
                    data = tomllib.load(fp)
            except Exception as e:
                print(f"[ERROR] reading {chosen}: {e}")
                continue

            epsg = self._extract_epsg_from_toml(data, self.default_epsg)
            marker_rows = self._extract_markers_from_deed(data, toml_spec)

            if not marker_rows:
                print(f"[INFO] No marker data found in: {chosen}")
                continue

            for r in marker_rows:
                r["File"] = file_prefix
                r["EPSG"] = epsg
                all_rows.append(r)

        if not all_rows:
            raise RuntimeError(
                "No marker data found in any TOML file (even though some TOMLs were found)."
            )

        final_columns = ["File"] + toml_spec + ["EPSG"]
        
        df_ID75 = pd.DataFrame(
            all_rows,
            columns=final_columns,
        )
        return df_ID75


# =========================================
# CoordinateTransformer 
# =========================================

class CoordinateTransformer:
    def __init__(self, crs_factory: CRSFactory):
        self.crs_factory = crs_factory
        self.col_northing = "NORTHING"
        self.col_easting = "EASTING"


    def to_wgs84(self, df_id75: pd.DataFrame) -> pd.DataFrame:
        """Indian 1975 (or other EPSG) → geographic WGS84 (EPSG:4326)."""
        lons = []
        lats = []
        for e, n, epsg in zip(
            df_id75[self.col_easting], df_id75[self.col_northing], df_id75["EPSG"]
        ):
            transformer = self.crs_factory.get_transformer_to_wgs84(int(epsg))
            lon, lat = transformer.transform(e, n)
            lons.append(lon)
            lats.append(lat)

        df_LL_W84 = df_id75.copy()
        df_LL_W84["LON"] = lons
        df_LL_W84["LAT"] = lats
        return df_LL_W84

    def to_w84_utm(self, df_id75: pd.DataFrame) -> pd.DataFrame:
        """
        Indian 1975 UTM (24047/24048) → WGS84 UTM (32647/32648).
        """
        xs = []
        ys = []
        epsg_out = []

        for e, n, epsg in zip(
            df_id75[self.col_easting], df_id75[self.col_northing], df_id75["EPSG"]
        ):
            epsg_src = int(epsg)
            transformer = self.crs_factory.get_transformer_to_w84_utm(epsg_src)
            x, y = transformer.transform(e, n)

            if epsg_src in (24047, 32647):
                epsg_dst = 32647
            elif epsg_src in (24048, 32648):
                epsg_dst = 32648
            else:
                epsg_dst = epsg_src

            xs.append(x)
            ys.append(y)
            epsg_out.append(epsg_dst)

        df_W84 = df_id75.copy()
        df_W84[self.col_easting] = xs
        df_W84[self.col_northing] = ys
        df_W84["EPSG"] = epsg_out
        return df_W84


# =========================================
# GPKG Writer 
# =========================================

class GPKGWriter:
    """Write three GPKG files: source CRS, geographic WGS84, WGS84 UTM."""

    def __init__(self, folder: Path, crs_factory: CRSFactory):
        self.folder = folder
        self.crs_factory = crs_factory
        self.col_northing = "NORTHING"
        self.col_easting = "EASTING"

    def write_ID75_W84(
        self,
        df_I75: pd.DataFrame,
        df_W84: pd.DataFrame,
        prefix: str,
    ):
        epsg_mode_src = int(df_I75["EPSG"].mode()[0])
        crs_i75utm = self.crs_factory.get_src_crs(epsg_mode_src)

        epsg_mode_w84utm = int(df_W84["EPSG"].mode()[0])
        crs_w84utm = CRS.from_epsg(epsg_mode_w84utm)

        # Use the provided prefix for GPKG filenames
        gpkg_i75utm_path = self.folder / f"{prefix}_I75UTM.gpkg"
        gpkg_w84utm_path = self.folder / f"{prefix}_W84UTM.gpkg"

        self.write_gpkg( df_I75, gpkg_i75utm_path, crs_i75utm )
        self.write_gpkg( df_W84, gpkg_w84utm_path, crs_w84utm )

    def write_gpkg(self, df: pd.DataFrame, gpkg_path, crs):
        for i, row in df.groupby('File'):
            print(f'Writing group {i} ...')
            
            # ---- marker points ----
            gdf_marker = gpd.GeoDataFrame(
                row.copy(),
                geometry=[Point(xy) for xy in zip(row[self.col_easting], row[self.col_northing])],
                crs=crs,
            )
            gdf_marker.to_file(gpkg_path, layer=f"marker:{i}", driver="GPKG")
            
            # ---- polygon boundary ----
            coords = list(zip(row[self.col_easting], row[self.col_northing]))
            if len(coords) > 1 and coords[0] != coords[-1]:
                coords.append(coords[0])
            
            boundary_geom = Polygon(coords)
            gdf_boundary = gpd.GeoDataFrame(
                {"File": [i]},
                geometry=[boundary_geom],
                crs=crs
                )
            gdf_boundary.to_file(gpkg_path, layer=f"parcel:{i}", driver="GPKG")
        print(f"[OK] Wrote GPKG → {gpkg_path}")


# =========================================
# High-level Processor 
# =========================================

class MarkerProcessor:
    """
    Orchestrates the whole flow:
    - Load config using Config_AppRV25J.
    - Load df_ID75 from folder.
    - Transform and write GPKG.
    - Archive source files using py7zr.
    """

    def __init__(
        self,
        folder: Path,
        config_path: Path,
        prefix: str, # Renamed from gpkg_prefix
    ):
        self.folder = folder
        self.config_path = config_path
        self.prefix = prefix # Renamed attribute
        
        # --- Load config using Config_AppRV25J ---
        app_file_dir = SCRIPT_DIR 
        working_folder = Path.cwd() 
        
        self.config_manager = Config_AppRV25J(
            app_file_dir=app_file_dir, 
            local_filename=self.config_path.name
        )
        
        # 1. Load Global Config (if it exists)
        log_msgs_global = self.config_manager.load_global_config_and_log()
        print("\n".join(log_msgs_global))
        
        # 2. Load Local Config (config.toml) and Merge
        log_msgs_local = self.config_manager.update_with_local(working_folder)
        print("\n".join(log_msgs_local))

        self.config_series = self.config_manager.CONFIG
        
        # Extract necessary values using utility functions
        self.default_epsg = _get_default_epsg(self.config_series)
        self.toml_spec = _get_toml_spec(self.config_series)
        towgs84 = _get_towgs84(self.config_series)
        
        print(f"[CONFIG] Default EPSG: {self.default_epsg}")
        print(f"[CONFIG] TOML_SPEC: {self.toml_spec}")
        print(f"[CONFIG] towgs84: {towgs84}")

        # Setup CRS factory
        self.crs_factory = CRSFactory(towgs84)


    def run(self):
        loader = MarkerLoader(
            self.folder, 
            self.toml_spec, 
            self.default_epsg
        )
        
        df_ID75 = loader.load_df_id75()

        # --- FULFILLMENT: List found TOML files ---
        print("\n=== Found Marker TOML Files ===")
        for f in loader.found_toml_files:
            print(f.resolve())
        # --------------------------------------------

        print("\n=== df_ID75 (source CRS) ===")
        print(df_ID75)

        transformer = CoordinateTransformer(self.crs_factory)

        df_LL_W84 = transformer.to_wgs84(df_ID75)
        print("\n=== df_LL_W84 (EPSG:4326) ===")
        # Print configured marker columns, plus LON/LAT (Note: self.toml_spec has 5 elements)
        print(df_LL_W84[["File"] + self.toml_spec[:-2] + ["LON", "LAT"]])

        df_W84 = transformer.to_w84_utm(df_ID75)
        print("\n=== df_W84 (WGS84 UTM; EPSG 32647/32648) ===")
        print(
            df_W84[
                ["File"] + self.toml_spec + ["EPSG"]
            ]
        )

        writer = GPKGWriter(self.folder, self.crs_factory)
        # Pass the renamed prefix attribute
        writer.write_ID75_W84(df_ID75, df_W84, self.prefix)
        
        # --- FULFILLMENT: Archive Requirement using py7zr (fixed) ---
        self._archive_toml_files(loader.found_toml_files)


    def _archive_toml_files(self, toml_files: List[Path]):
        """
        Archives the list of *_OCRedit.toml files into {PREFIX}_deed2cadas.7z 
        or deed2cadas.7z if no prefix is used, saving it to self.folder.
        """
        if not toml_files:
            print("\n[ARCHIVE] No *_OCRedit.toml files found to archive. Skipping.")
            return
        
        # Logic to determine the archive name based on the prefix (Request 2)
        if self.prefix == "cadastre": # Check against the default value defined in argparse
            archive_name = "deed2cadas.7z"
        else:
            archive_name = f"{self.prefix}_deed2cadas.7z"

        # Explicitly set the output path to the input folder (self.folder) (Request to put beside GPKGs)
        archive_path = self.folder / archive_name
        
        print(f"\n[ARCHIVE] Creating 7zip archive: {archive_name} in folder: {self.folder.resolve()}...")

        try:
            with py7zr.SevenZipFile(archive_path, 'w') as archive:
                for file_path in toml_files:
                    # Use .name as the archive name for the content inside the 7z file, 
                    # and str(file_path) as the actual path to the file on disk.
                    if file_path.is_file():
                        archive.write(str(file_path), file_path.name)
                    else:
                        print(f"[WARNING] Skipping missing file: {file_path}")

            if archive_path.is_file():
                print(f"[ARCHIVE] Successfully created {archive_name} containing {len(toml_files)} file(s).")
            else:
                 print(f"[ERROR] Archive file {archive_name} was not created or is empty.")
            
        except Exception as e:
            print(f"\n[ERROR] An error occurred during 7zip archiving using py7zr: {e}")
            print(f"[ERROR] Cannot create archive: {archive_name}")


# =========================================
# main() (Updated argparse)
# =========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RV25J Cadastre Marker Processor (CONFIG.toml required, ID→WGS84/W84UTM)"
    )
    parser.add_argument(
        "folder",
        help="Root folder containing *_OCRedit.toml (recursively).",
    )
    # Renamed argument from --gpkg-prefix to --prefix (Request 1)
    parser.add_argument(
        "--prefix",
        default="cadastre",
        help="Prefix for output GPKG files and the 7zip archive (default: 'cadastre').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path("config.toml") 
    if not config_path.is_file():
        print(f"[ERROR] {config_path.name} not found — must exist in current directory.")
        sys.exit(1)

    folder = Path(args.folder)
    if not folder.is_dir():
         print(f"[ERROR] Input folder not found: {folder}")
         sys.exit(1)
         
    try:
        processor = MarkerProcessor(
            folder=folder,
            config_path=config_path,
            prefix=args.prefix, # Pass the renamed argument
        )
        processor.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] Application failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()