# =============================================================================
# Tropical Weather & Climate Observatory (TWCO)
# Product: ECMWF IFS Tropical Forecast Precipitation | 6-hour accumulations
#
# Same regions as the IMERG product:
#   1) Global Tropics
#   2) Americas
#   3) Mexico
#   4) Colombia
#
# Download IFS global TP fields once; retain tropical belt for regional maps.
# IFS latitude/longitude coordinates are CELL CENTERS.
# Convert cell centers to EDGES before pcolormesh(shading="flat").
# Output: 32 lossless WebP frames and a JSON manifest.
#
# Author: Alejandro Jaramillo Moreno
# Website: https://ajaramillomoreno.github.io
# =============================================================================

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import io
import json
import os
import tempfile

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from PIL import Image
from ecmwf.opendata import Client

from eccodes import (
    codes_grib_new_from_file, codes_get, codes_get_array,
    codes_get_values, codes_release,
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "forecast_ecmwf"
JSON = OUTPUT / "ecmwf_tropical_precip_forecast_latest.json"

parser = argparse.ArgumentParser(
    description="Generate multi-region tropical ECMWF IFS precipitation forecast frames."
)
parser.add_argument(
    "--output-dir", type=Path, default=None,
    help="Alternative destination, e.g. sandbox/ecmwf_tropical_test"
)
parser.add_argument(
    "--force", action="store_true",
    help="Generate again even if this IFS cycle is already published"
)
args = parser.parse_args()

if args.output_dir is not None:
    OUTPUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    JSON = OUTPUT / "ecmwf_tropical_precip_forecast_latest.json"

OUTPUT.mkdir(parents=True, exist_ok=True)

cartopy.config["data_dir"] = ROOT / ".cache" / "cartopy"
Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

ECMWF_SOURCE = "ecmwf"  # Alternatives: "aws", "google", "azure"
FORECAST_HOURS = list(range(6, 49, 6))
LAT_MIN, LAT_MAX = -35, 35

# Same regions, extents, and figure sizes used in the IMERG product.
MAPS = [
    ("global_tropics", "ECMWF IFS Tropical Forecast Precipitation",
     [-180, 180, -35, 35], (18, 6.4), 60, 15),
    ("americas", "ECMWF IFS Forecast Precipitation over the Americas",
     [-120, -30, -35, 35], (12, 8), 20, 10),
    ("mexico", "ECMWF IFS 6-hour Forecast Precipitation over Mexico",
     [-118, -86, 14, 33], (12, 7), 5, 5),
    ("colombia", "ECMWF IFS 6-hour Forecast Precipitation over Colombia",
     [-84, -66, -6, 15], (9, 9), 4, 5),
]

# Six-hour precipitation palette.
PRECIP_LEVELS = [0.5, 1, 2.5, 5, 10, 20, 35, 50, 75, 100, 150]
PRECIP_COLORS = [
    "#c8f0e7", "#8ce1c8", "#43c99b", "#15aa75", "#4cbb46",
    "#d1d93b", "#f5b63b", "#f47b36", "#e84646", "#a8327b", "#592b86",
]
cmap = ListedColormap(PRECIP_COLORS)
cmap.set_under("#ffffff")
cmap.set_over("#34144f")
# Transparent bad values avoids Cartopy warnings on the global map.
cmap.set_bad((0, 0, 0, 0))
norm = BoundaryNorm(PRECIP_LEVELS, cmap.N)

# =============================================================================
# 2. ADMINISTRATIVE BOUNDARIES
# =============================================================================

admin1_path = shapereader.natural_earth(
    resolution="10m", category="cultural",
    name="admin_1_states_provinces",
)
admin1_records = list(shapereader.Reader(admin1_path).records())

mexico_admin1 = [
    rec.geometry for rec in admin1_records
    if rec.attributes.get("admin") == "Mexico"
]
colombia_admin1 = [
    rec.geometry for rec in admin1_records
    if rec.attributes.get("admin") == "Colombia"
]

if not mexico_admin1 or not colombia_admin1:
    raise RuntimeError("Mexican states or Colombian departments not found in Natural Earth.")

# =============================================================================
# 3. HELPER: SAVE FIGURE AS LOSSLESS WEBP
# =============================================================================

def save_lossless_webp(fig, destination, dpi=200):
    """Render the Matplotlib figure to PNG in memory and convert it to lossless WebP."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.load()
        image.save(destination, "WEBP", lossless=True, method=6)
    buffer.close()

# =============================================================================
# 4. DOWNLOAD THE LATEST COMPLETE IFS FORECAST
# =============================================================================
# ECMWF IFS operational precipitation is normally accumulated from forecast
# initialization. We inspect GRIB startStep/endStep before calculating 6-h totals.
# If the source ever changes to independent 6-h windows, the reader handles that.

lat = np.arange(LAT_MIN, LAT_MAX + 0.125, 0.25)
lon = np.arange(-180.0, 180.0, 0.25)


def read_precipitation_grib(path, cycle):
    """Return {lead: (start_step, tropical_precip_mm)} from an IFS GRIB2 file."""
    records = {}
    with path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                if str(codes_get(gid, "shortName")).lower() != "tp":
                    continue
                if str(codes_get(gid, "stepType")) != "accum":
                    raise ValueError("ECMWF TP is not an accumulated field")

                start = int(codes_get(gid, "startStep"))
                end = int(codes_get(gid, "endStep"))
                step_units = str(codes_get(gid, "stepUnits")).lower()
                if step_units not in ("h", "1", "hour", "hours"):
                    raise ValueError(f"Unexpected GRIB forecast-time units: {step_units}")
                if end not in FORECAST_HOURS or start < 0 or start >= end:
                    raise ValueError(f"Unexpected TP accumulation window: {start}-{end} h")
                if end in records:
                    raise ValueError(f"Duplicate TP message at step {end}")
                if int(codes_get(gid, "dataDate")) != int(cycle.strftime("%Y%m%d")):
                    raise ValueError("GRIB initialization date differs from requested cycle")
                if int(codes_get(gid, "dataTime")) != int(cycle.strftime("%H%M")):
                    raise ValueError("GRIB initialization time differs from requested cycle")

                units = str(codes_get(gid, "units")).lower().strip()
                if units in ("m", "metres", "meters"):
                    multiplier = 1000.0  # metres of water -> mm
                elif units in ("kg m**-2", "kg m-2", "kg/m^2", "mm"):
                    multiplier = 1.0
                else:
                    raise ValueError(f"Unsupported precipitation units: {units}")

                values = np.asarray(codes_get_values(gid), dtype=np.float32)
                lats = np.asarray(codes_get_array(gid, "latitudes"), dtype=np.float32)
                lons = np.asarray(codes_get_array(gid, "longitudes"), dtype=np.float32)
                if not (values.size == lats.size == lons.size):
                    raise ValueError("Inconsistent ECMWF grid coordinates")

                lons = np.where(lons >= 180, lons - 360, lons)
                mask = (lats >= LAT_MIN - 1e-4) & (lats <= LAT_MAX + 1e-4)
                y_float = (lats[mask] - LAT_MIN) / 0.25
                x_float = (lons[mask] + 180.0) / 0.25
                if not np.allclose(y_float, np.rint(y_float), atol=1e-3):
                    raise ValueError("ECMWF latitude grid differs from 0.25 degrees")
                if not np.allclose(x_float, np.rint(x_float), atol=1e-3):
                    raise ValueError("ECMWF longitude grid differs from 0.25 degrees")

                iy = np.rint(y_float).astype(int)
                ix = np.rint(x_float).astype(int)
                field = np.full((lat.size, lon.size), np.nan, dtype=np.float32)
                field[iy, ix] = values[mask] * multiplier
                if not np.isfinite(field).all():
                    raise ValueError("ECMWF TP does not cover all tropical grid cells")
                if field.min() < -0.1 or field.max() > 2000:
                    raise ValueError("Unreasonable ECMWF TP values")

                records[end] = (start, field)
                print(f"  +{end:03d} h | GRIB period {start}-{end} h | max {field.max():.1f} mm")
            finally:
                codes_release(gid)

    if set(records) != set(FORECAST_HOURS):
        raise ValueError(f"Missing TP forecast steps: {sorted(set(FORECAST_HOURS) - set(records))}")
    return records


def six_hour_fields(records):
    """Convert ECMWF GRIB accumulated precipitation into independent 6-h totals."""
    windows = [records[lead][0] for lead in FORECAST_HOURS]
    cumulative = all(start == 0 for start in windows)
    independent = all(start == lead - 6 for start, lead in zip(windows, FORECAST_HOURS))
    if not (cumulative or independent):
        raise ValueError(f"Unexpected/inconsistent ECMWF accumulation windows: {windows}")

    fields = []
    previous = np.zeros_like(records[FORECAST_HOURS[0]][1])
    for lead in FORECAST_HOURS:
        current = records[lead][1]
        precip = current - previous if cumulative else current.copy()
        # Very small negative differences can arise from GRIB rounding.
        if np.min(precip) < -0.5:
            raise ValueError(f"Negative 6-hour precipitation at +{lead} h: {precip.min():.2f} mm")
        fields.append(np.maximum(precip, 0))
        previous = current
    return fields, "cumulative_from_start" if cumulative else "independent_6h"


# The IFS data arrive several hours after model initialization. Start one cycle
# behind current UTC, then fall back to earlier runs if any lead is unavailable.
now = datetime.now(timezone.utc) - timedelta(hours=6)
latest_hour = (now.hour // 6) * 6
latest_cycle = now.replace(hour=latest_hour, minute=0, second=0, microsecond=0)
cycles = [latest_cycle - timedelta(hours=6 * i) for i in range(8)]
client = Client(source=ECMWF_SOURCE, model="ifs", resol="0p25")

selected_cycle = None
forecast_fields = None
accumulation_type = None

print("TWCO | ECMWF IFS Tropical Forecast Precipitation (0.25°)")
print(f"Script path: {Path(__file__).resolve()}")
print(f"Output directory: {OUTPUT}")
print("Forecast: independent 6-hour accumulations from +0 to +48 h")
print("Grid: 35°S–35°N, all longitudes")

with tempfile.TemporaryDirectory(prefix="twco_ifs_tropical_") as temp_dir:
    grib_file = Path(temp_dir) / "ifs_tp.grib2"
    for cycle in cycles:
        print(f"\nChecking ECMWF IFS cycle {cycle:%Y-%m-%d %H}:00 UTC...")
        try:
            client.retrieve(
                date=cycle.strftime("%Y%m%d"),
                time=cycle.hour,
                stream="oper",
                type="fc",
                levtype="sfc",
                param="tp",
                step=FORECAST_HOURS,
                target=str(grib_file),
            )
            if not grib_file.is_file() or grib_file.stat().st_size < 1000:
                raise ValueError("Downloaded ECMWF GRIB file is missing or empty")

            records = read_precipitation_grib(grib_file, cycle)
            forecast_fields, accumulation_type = six_hour_fields(records)
            selected_cycle = cycle
            break
        except (Exception) as error:
            # Network failures and unavailable runs should not prevent fallback.
            print(f"  Cycle incomplete or incompatible: {str(error)[:240]}")
            grib_file.unlink(missing_ok=True)

if selected_cycle is None:
    raise RuntimeError("No complete ECMWF IFS cycle with +6 to +48 h TP was available")

print(f"\nSelected IFS cycle: {selected_cycle:%Y-%m-%d %H}:00 UTC")
print(f"GRIB accumulation encoding: {accumulation_type}")

# Never replace an equal or newer published product unless --force is used.
if JSON.exists() and not args.force:
    try:
        previous = json.loads(JSON.read_text(encoding="utf-8"))
        previous_cycle = datetime.fromisoformat(previous["ecmwf_cycle_utc"])
        if previous_cycle >= selected_cycle:
            print("Already published this IFS cycle or a newer one. No change.")
            raise SystemExit(0)
    except (KeyError, ValueError, json.JSONDecodeError):
        print("Previous metadata cannot be read; regenerating.")

# =============================================================================
# 5. GRID GEOMETRY: IFS CELL CENTERS -> PIXEL EDGES
# =============================================================================
# The IFS GRIB coordinates identify each cell CENTER.
# For pcolormesh(shading="flat"), we explicitly construct the CELL EDGES.
# For the regular 0.25-degree grid, each pixel extends +/- 0.125 degrees.
# Deriving borders from midpoints also works if the grid spacing changes.

if lat[0] > lat[-1]:
    lat = lat[::-1]
    forecast_fields = [field[::-1, :] for field in forecast_fields]

lon_order = np.argsort(lon)
lon = lon[lon_order]
forecast_fields = [field[:, lon_order] for field in forecast_fields]

lon_edges = np.r_[
    lon[0] - (lon[1] - lon[0]) / 2,
    (lon[:-1] + lon[1:]) / 2,
    lon[-1] + (lon[-1] - lon[-2]) / 2,
]
lat_edges = np.r_[
    lat[0] - (lat[1] - lat[0]) / 2,
    (lat[:-1] + lat[1:]) / 2,
    lat[-1] + (lat[-1] - lat[-2]) / 2,
]

if lon_edges.size != lon.size + 1 or lat_edges.size != lat.size + 1:
    raise RuntimeError("The IFS cell edges do not match the grid centers")

# =============================================================================
# 6. RENDER AND PUBLISH
# =============================================================================

plt.rcParams["font.family"] = "DejaVu Sans"
expected_frames = len(MAPS) * len(FORECAST_HOURS)

with tempfile.TemporaryDirectory(prefix=".twco_publish_", dir=OUTPUT) as publish_tmp:
    publish_tmp = Path(publish_tmp)

    metadata = {
        "product": "TWCO-ECMWF-IFS-TROPICAL",
        "title": "ECMWF IFS Tropical Forecast Precipitation",
        "source": "ECMWF IFS Open Data 0.25°",
        "ecmwf_cycle_utc": selected_cycle.isoformat(),
        "forecast_hours": FORECAST_HOURS,
        "interval_hours": 6,
        "domain": "35S-35N, all longitudes",
        "units": "mm / 6 h",
        "image_format": "webp",
        "compression": "lossless",
        "grid_resolution_degrees": 0.25,
        "pixel_coordinates": "IFS centers converted to cell edges for pcolormesh(shading=flat)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regions": [],
        "data_source_url": "https://data.ecmwf.int/forecasts/",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "parameter": "tp",
        "accumulation_encoding": accumulation_type,
    }

    generated_count = 0

    for key, title, extent, figsize, lon_step, lat_step in MAPS:
        print(f"\nRendering region: {key}")
        region_dir = publish_tmp / "frames" / key
        region_dir.mkdir(parents=True, exist_ok=True)

        region_info = {
            "key": key,
            "title": title,
            "extent": extent,
            "figure_size_inches": list(figsize),
            "frames": [],
        }

        # Select the IFS subregion once. No need to send the whole global
        # matrix to Cartopy for the Americas, Mexico or Colombia.
        # Include a one-cell margin at regional borders.
        if key == "global_tropics":
            iy0, iy1 = 0, lat.size
            ix0, ix1 = 0, lon.size
        else:
            pad = 0.25  # one IFS grid cell
            iy = np.flatnonzero((lat >= extent[2] - pad) & (lat <= extent[3] + pad))
            ix = np.flatnonzero((lon >= extent[0] - pad) & (lon <= extent[1] + pad))
            if iy.size < 2 or ix.size < 2:
                raise RuntimeError(f"Insufficient IFS grid cells for {key}")
            iy0, iy1 = int(iy[0]), int(iy[-1]) + 1
            ix0, ix1 = int(ix[0]), int(ix[-1]) + 1

        region_lat_edges = lat_edges[iy0:iy1 + 1]
        region_lon_edges = lon_edges[ix0:ix1 + 1]
        # A field with (ny, nx) cells needs (ny+1, nx+1) pixel edges.
        if region_lat_edges.size != (iy1 - iy0 + 1) or region_lon_edges.size != (ix1 - ix0 + 1):
            raise RuntimeError(f"Pixel edges do not match the {key} crop")
        print(f"  IFS grid for map: {iy1 - iy0} x {ix1 - ix0} cells")

        for lead, field in zip(FORECAST_HOURS, forecast_fields):
            start_hour = lead - 6
            period_start = selected_cycle + timedelta(hours=start_hour)
            period_end = selected_cycle + timedelta(hours=lead)

            fig = plt.figure(figsize=figsize, facecolor="white")
            ax = fig.add_axes([0.09, 0.28, 0.82, 0.52], projection=ccrs.PlateCarree())

            # Crop only the plotting array; the original global forecast
            # remains unchanged for all other regions and forecast times.
            regional_field = field[iy0:iy1, ix0:ix1]
            if regional_field.shape != (region_lat_edges.size - 1, region_lon_edges.size - 1):
                raise RuntimeError(f"Regional precipitation and cell edges disagree for {key}")
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            mesh = ax.pcolormesh(
                region_lon_edges, region_lat_edges,
                np.ma.masked_less(regional_field, PRECIP_LEVELS[0]),
                transform=ccrs.PlateCarree(), shading="flat",
                cmap=cmap, norm=norm, rasterized=True
            )

            if key == "global_tropics":
                ax.coastlines(resolution="110m", color="#273847", linewidth=0.65)
            else:
                ax.coastlines(resolution="10m", color="#273847", linewidth=0.8)
                ax.add_feature(
                    cfeature.BORDERS.with_scale("10m"),
                    edgecolor="#273847", linewidth=0.85, zorder=5,
                )
                if key == "mexico":
                    ax.add_geometries(
                        mexico_admin1, crs=ccrs.PlateCarree(),
                        facecolor="none", edgecolor="#404c59",
                        linewidth=0.45, alpha=0.9, zorder=6,
                    )
                elif key == "colombia":
                    ax.add_geometries(
                        colombia_admin1, crs=ccrs.PlateCarree(),
                        facecolor="none", edgecolor="#404c59",
                        linewidth=0.55, alpha=0.9, zorder=6,
                    )

            ax.set_xticks(
                np.arange(np.ceil(extent[0] / lon_step) * lon_step, extent[1] + 0.001, lon_step),
                crs=ccrs.PlateCarree(),
            )
            ax.set_yticks(
                np.arange(np.ceil(extent[2] / lat_step) * lat_step, extent[3] + 0.001, lat_step),
                crs=ccrs.PlateCarree(),
            )
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.gridlines(linewidth=0.4, alpha=0.3, color="#555555", linestyle=":")
            ax.tick_params(labelsize=9)

            fig.text(0.055, 0.95, "TROPICAL WEATHER & CLIMATE OBSERVATORY",
                     fontsize=11, weight="bold", color="#1c405b")
            fig.text(0.055, 0.885, title,
                     fontsize=19, weight="bold", color="#152a3c")
            fig.text(
                0.055, 0.83,
                f"Initialized: {selected_cycle:%d %b %Y %H}:00 UTC  |  "
                f"Forecast interval: +{start_hour:03d} to +{lead:03d} h",
                fontsize=10, color="#475569"
            )
            # Keep validity below the map, with extra separation for regional maps.
            valid_y = 0.245 if key == "global_tropics" else 0.225
            fig.text(
                0.50, valid_y,
                f"Valid: {period_start:%d %b %Y %H}:00 – {period_end:%d %b %Y %H}:00 UTC",
                fontsize=10, color="#475569", ha="center", va="center"
            )

            if key == "global_tropics":
                cax = fig.add_axes([0.15, 0.185, 0.70, 0.032])
                cb = fig.colorbar(
                    mesh, cax=cax, orientation="horizontal",
                    extend="max", ticks=PRECIP_LEVELS,
                )
                cb.set_label("Accumulated precipitation (mm / 6 h)", fontsize=11)
                cb.ax.tick_params(labelsize=9)
            else:
                # Let Cartopy finalize map aspect before sizing the colorbar.
                fig.canvas.draw()
                map_box = ax.get_position()
                bar_width = 0.90 * map_box.width
                bar_left = map_box.x0 + (map_box.width - bar_width) / 2
                bar_height = 0.016
                cax = fig.add_axes([bar_left, 0.185, bar_width, bar_height])
                regional_ticks = [0.5, 2.5, 10, 35, 75, 150]
                cb = fig.colorbar(
                    mesh, cax=cax, orientation="horizontal",
                    extend="max", ticks=regional_ticks,
                )
                cb.set_label("Accumulated precipitation (mm / 6 h)", fontsize=10)
                cb.ax.tick_params(labelsize=8, length=2, pad=3)

            fig.text(
                0.055, 0.065,
                "Source: ECMWF IFS Open Data 0.25° (CC BY 4.0)  |  TP  |  6-hour totals",
                fontsize=9, color="#64748b"
            )
            fig.text(
                0.945, 0.065,
                "ajaramillomoreno.github.io",
                fontsize=9, color="#64748b", ha="right"
            )

            filename = f"ecmwf_precip_f{lead:03d}.webp"
            destination = region_dir / filename
            save_lossless_webp(fig, destination, dpi=200)
            plt.close(fig)
            generated_count += 1

            region_info["frames"].append({
                "forecast_hour": lead,
                "interval_start_hour": start_hour,
                "interval_end_hour": lead,
                "start_utc": period_start.isoformat(),
                "end_utc": period_end.isoformat(),
                "image": f"frames/{key}/{filename}",
            })

        metadata["regions"].append(region_info)

    if generated_count != expected_frames:
        raise RuntimeError(
            f"Unexpected number of generated frames: {generated_count} (expected {expected_frames})"
        )
    # Verify all expected files exist *before* publishing any file.
    for key, *_ in MAPS:
        for lead in FORECAST_HOURS:
            source = publish_tmp / "frames" / key / f"ecmwf_precip_f{lead:03d}.webp"
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty forecast frame: {source}")

    (publish_tmp / JSON.name).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Publish only the expected frames (avoids macOS hidden files like ._...)
    for key, *_ in MAPS:
        src_dir = publish_tmp / "frames" / key
        dst_dir = OUTPUT / "frames" / key
        dst_dir.mkdir(parents=True, exist_ok=True)

        for lead in FORECAST_HOURS:
            filename = f"ecmwf_precip_f{lead:03d}.webp"
            source = src_dir / filename
            target = dst_dir / filename

            if not source.is_file():
                raise RuntimeError(f"Missing forecast frame: {source}")

            os.replace(source, target)
            print(f"Saved: {target}")

    os.replace(publish_tmp / JSON.name, JSON)
    print(f"Metadata saved: {JSON}")

print(f"\nDone. {expected_frames} lossless WebP frames and JSON manifest generated.")
