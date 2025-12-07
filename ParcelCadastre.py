#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV25j_Cadastre.py — RV25J Marker Processor (OOP, CONFIG.toml required)

Assumptions
-----------
1) CONFIG.toml อยู่ใน current working directory เสมอ และมีโครงแบบ:
   [META]
   DOL_Office = "Narathivas"
   towgs84 = [204.5,837.9,294.8]
   TOML_SPEC = [ "SEQ_NUM", "MRK_SEQ", "MRK_DOL" ,"NORTHING","EASTING" ]

2) Marker source files are now *_OCRedit.toml, containing the [Deed].marker array.
   marker = [
     [1, "A", "s24", 711494.218, 810313.001],
     ...
   ]
   Interpreted as:
     [SEQ_NUM, MRK_SEQ, MRK_DOL, NORTHING, EASTING]

3) Workflow
   - ... (Transformation and GPKG writing) ...
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from pyproj import CRS, Transformer

# --- TOML loader ---
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # fallback for older versions


# =========================================
# Config class
# =========================================

class RV25JConfig:
    """Read and store values from CONFIG.toml"""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

        meta = data.get("META", {})
        deed = data.get("Deed", {}) or data.get("deed", {})
        center = data.get("RV25J_CENTER", {})

        self.dol_office = meta.get("DOL_Office")
        self.towgs84 = meta.get("towgs84", None)  # list [dx, dy, dz]
        self.view_scale = center.get("view_scale", 0.5)
        # NEW: Load TOML_SPEC
        self.toml_spec = meta.get("TOML_SPEC", [ "SEQ_NUM", "MRK_SEQ", "MRK_DOL" ,"NORTHING","EASTING" ])

        # Determine default EPSG from [Deed].EPSG or [Deed].crs
        epsg = deed.get("EPSG") or deed.get("epsg")
        if epsg is None:
            crs_val = deed.get("crs") or deed.get("CRS")
            if crs_val is not None:
                try:
                    epsg = int(crs_val)
                except ValueError:
                    epsg = None
        if epsg is None:
            epsg = 24047  # fallback
        self.default_epsg = int(epsg)

    @classmethod
    def from_toml(cls, config_path: Path) -> "RV25JConfig":
        if not config_path.is_file():
            raise FileNotFoundError(f"CONFIG.toml not found: {config_path}")
        with config_path.open("rb") as fp:
            data = tomllib.load(fp)
        return cls(config_path, data)

    def __repr__(self):
        return (
            f"RV25JConfig(DOL_Office={self.dol_office!r}, "
            f"towgs84={self.towgs84}, "
            f"view_scale={self.view_scale}, "
            f"default_epsg={self.default_epsg})"
        )


# =========================================
# CRS / Transformer factory (No changes)
# =========================================
# ... (CRSFactory class remains unchanged)
class CRSFactory:
    """
    Build CRS for Indian 1975 UTM (EPSG 24047 / 24048) with towgs84 if provided,
    or fallback to standard EPSG (e.g. 32647).
    """

    def __init__(self, towgs84: List[float] | None):
        self.towgs84 = towgs84
        self._crs_cache: Dict[int, CRS] = {}
        self._transformer_cache: Dict[int, Transformer] = {}
        self._crs_wgs84 = CRS.from_epsg(4326)

        # cache for WGS84 UTM (output) CRSs and transformers
        self._crs_w84_utm_cache: Dict[int, CRS] = {}
        self._transformer_w84_utm_cache: Dict[int, Transformer] = {}

    def _build_proj4_id75(self, epsg: int) -> CRS:
        """
        For EPSG 24047/24048, build Indian 1975 / UTM zone 47 or 48
        with ellipsoid + towgs84. Otherwise use EPSG directly (e.g. 32647).
        """
        if epsg == 24047:
            zone = 47
        elif epsg == 24048:
            zone = 48
        else:
            # Non-Indian 1975: use EPSG directly (e.g. 32647)
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
        """Return CRS for given EPSG (ID75 w/ towgs84 or normal EPSG)."""
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
        """
        Map ID75 (24047/24048) or UTM WGS84 (32647/32648) to WGS84 UTM CRS.
        24047,32647 -> EPSG:32647
        24048,32648 -> EPSG:32648
        others     -> keep same EPSG as fallback.
        """
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
        """
        Transformer from source CRS (ID75 / existing EPSG) to WGS84 UTM.
        """
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
    - Build df_ID75 with columns defined by TOML_SPEC + File + EPSG
    """

    def __init__(self, folder: Path, config: RV25JConfig):
        self.folder = folder
        self.config = config

    @staticmethod
    def _file_prefix_from_path(path: Path) -> str:
        """
        'p08_OCRedit.toml' -> 'p08'
        """
        stem = path.stem  # e.g. "p08_OCRedit"
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
        using column names from toml_spec.
        """
        rows = []

        deed = toml_data.get("Deed") or toml_data.get("deed")
        if not isinstance(deed, dict):
            return rows

        marker_arr = deed.get("marker")
        if not isinstance(marker_arr, list):
            return rows

        # Check if TOML_SPEC has the expected 5 elements
        if len(toml_spec) < 5:
            print(f"[ERROR] TOML_SPEC has less than 5 elements. Cannot map marker array.")
            return rows

        for entry in marker_arr:
            # Marker array from OCRedit.toml has 5 elements
            if not isinstance(entry, (list, tuple)) or len(entry) < 5:
                continue

            # Unpack based on the expected fixed position of data in the TOML list
            idx_raw, marker_raw, code_raw, n_raw, e_raw = entry[:5]

            try:
                n_val = float(n_raw)
                e_val = float(e_raw)
            except Exception:
                continue

            # Map the raw data to the column names specified in TOML_SPEC
            rows.append(
                {
                    toml_spec[0]: idx_raw,       # SEQ_NUM (Index 0)
                    toml_spec[1]: marker_raw,    # MRK_SEQ (Index 1)
                    toml_spec[2]: code_raw,      # MRK_DOL (Index 2)
                    toml_spec[3]: n_val,         # NORTHING (Index 3)
                    toml_spec[4]: e_val,         # EASTING (Index 4)
                }
            )

        return rows

    def load_df_id75(self) -> pd.DataFrame:
        """
        Only searches for *_OCRedit.toml.
        Returns df_ID75 with columns: TOML_SPEC + ["File", "EPSG"]
        """
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Folder not found: {self.folder}")

        # Recursive search for *_OCRedit.toml
        toml_files = list(self.folder.rglob("*_OCRedit.toml"))

        if not toml_files:
            raise FileNotFoundError(
                f"No *_OCRedit.toml files found under {self.folder}"
            )

        all_rows = []
        toml_spec = self.config.toml_spec
        
        for chosen in toml_files:
            file_prefix = self._file_prefix_from_path(chosen)

            try:
                with chosen.open("rb") as fp:
                    data = tomllib.load(fp)
            except Exception as e:
                print(f"[ERROR] reading {chosen}: {e}")
                continue

            epsg = self._extract_epsg_from_toml(data, self.config.default_epsg)
            marker_rows = self._extract_markers_from_deed(data, toml_spec) # Pass TOML_SPEC

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

        # Columns: ["File"] + TOML_SPEC + ["EPSG"]
        final_columns = ["File"] + toml_spec + ["EPSG"]
        
        df_ID75 = pd.DataFrame(
            all_rows,
            columns=final_columns,
        )
        return df_ID75


# =========================================
# CoordinateTransformer (Adjust column access)
# =========================================

class CoordinateTransformer:
    """Use CRSFactory to transform coordinates."""

    def __init__(self, crs_factory: CRSFactory):
        self.crs_factory = crs_factory
        # Assuming TOML_SPEC is [ ..., NORTHING, EASTING]
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
                epsg_dst = epsg_src  # fallback

            xs.append(x)
            ys.append(y)
            epsg_out.append(epsg_dst)

        df_W84 = df_id75.copy()
        # Overwrite/Add columns with WGS84 UTM values
        df_W84[self.col_easting] = xs
        df_W84[self.col_northing] = ys
        df_W84["EPSG"] = epsg_out
        return df_W84


# =========================================
# GPKG Writer (Adjust column access in write_gpkg)
# =========================================

class GPKGWriter:
    """Write three GPKG files: source CRS, geographic WGS84, WGS84 UTM."""

    def __init__(self, folder: Path, crs_factory: CRSFactory):
        self.folder = folder
        self.crs_factory = crs_factory
        # Assuming TOML_SPEC is [ ..., NORTHING, EASTING]
        self.col_northing = "NORTHING"
        self.col_easting = "EASTING"

    def write_ID75_W84(
        self,
        df_I75: pd.DataFrame,
        df_W84: pd.DataFrame,
        prefix: str,
    ):
        # Use mode of EPSG as representative CRS for each output
        epsg_mode_src = int(df_I75["EPSG"].mode()[0])
        crs_i75utm = self.crs_factory.get_src_crs(epsg_mode_src)

        epsg_mode_w84utm = int(df_W84["EPSG"].mode()[0])
        crs_w84utm = CRS.from_epsg(epsg_mode_w84utm)

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
                # Use the configured column names for coordinates
                geometry=[Point(xy) for xy in zip(row[self.col_easting], row[self.col_northing])],
                crs=crs,
            )
            gdf_marker.to_file(gpkg_path, layer=f"marker:{i}", driver="GPKG")
            # ---- polygon boundary ----
            coords = list(zip(row[self.col_easting], row[self.col_northing]))
            # ensure closed ring
            if len(coords) > 1 and coords[0] != coords[-1]:
                coords.append(coords[0])
            # create Polygon instead of LineString
            boundary_geom = Polygon(coords)
            gdf_boundary = gpd.GeoDataFrame(
                {"File": [i]},
                geometry=[boundary_geom],
                crs=crs
                )
            gdf_boundary.to_file(gpkg_path, layer=f"parcel:{i}", driver="GPKG")
        print(f"[OK] Wrote GPKG → {gpkg_path}")


# =========================================
# High-level Processor (Adjust print columns)
# =========================================

class MarkerProcessor:
    """
    Orchestrates the whole flow:
    - Load CONFIG.toml
    - Load df_ID75 from folder
    - Transform to df_LL_W84 and df_W84
    - Optional CSV (df_ID75)
    - Write GPKG (ID, WGS84, W84UTM)
    """

    def __init__(
        self,
        folder: Path,
        config_path: Path,
        gpkg_prefix: str,
    ):
        self.folder = folder
        self.config_path = config_path
        self.gpkg_prefix = gpkg_prefix

        # Load config
        self.config = RV25JConfig.from_toml(config_path)
        print(f"[CONFIG] {self.config}")

        # Setup CRS factory
        self.crs_factory = CRSFactory(self.config.towgs84)

    def run(self):
        loader = MarkerLoader(self.folder, self.config)
        df_ID75 = loader.load_df_id75()
        print("\n=== df_ID75 (source CRS) ===")
        print(df_ID75) # Print all columns for source CRS DF

        transformer = CoordinateTransformer(self.crs_factory)

        # Geographic WGS84
        df_LL_W84 = transformer.to_wgs84(df_ID75)
        print("\n=== df_LL_W84 (EPSG:4326) ===")
        # Print configured marker columns, plus LON/LAT
        print(df_LL_W84[["File"] + self.config.toml_spec[:-2] + ["LON", "LAT"]])

        # WGS84 UTM
        df_W84 = transformer.to_w84_utm(df_ID75)
        print("\n=== df_W84 (WGS84 UTM; EPSG 32647/32648) ===")
        # Print all configured columns plus EPSG
        print(
            df_W84[
                ["File"] + self.config.toml_spec + ["EPSG"]
            ]
        )

        writer = GPKGWriter(self.folder, self.crs_factory)
        writer.write_ID75_W84(df_ID75, df_W84, self.gpkg_prefix)


# =========================================
# main() (No changes)
# =========================================
# ... (main function remains unchanged)
def parse_args():
    parser = argparse.ArgumentParser(
        description="RV25J Cadastre Marker Processor (CONFIG.toml required, ID→WGS84/W84UTM)"
    )
    # positional: folder
    parser.add_argument(
        "folder",
        help="Root folder containing *_OCRedit.toml (recursively).",
    )
    parser.add_argument(
        "--gpkg-prefix",
        default="cadastre",
        help="Prefix for output GPKG files (default: 'cadastre').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # CONFIG.toml must exist in current directory
    config_path = Path("CONFIG.toml")
    if not config_path.is_file():
        print("[ERROR] CONFIG.toml not found — must exist in current directory.")
        sys.exit(1)

    folder = Path(args.folder)
    processor = MarkerProcessor(
        folder=folder,
        config_path=config_path,
        gpkg_prefix=args.gpkg_prefix,
    )
    processor.run()


if __name__ == "__main__":
    main()