# =============================================================================
# Tropical Weather & Climate Observatory (TWCO)
# Product: GFS Tropical Forecast Precipitation | 6-hour accumulations
#
# Same regions as the IMERG product:
#   1) Global Tropics
#   2) Americas
#   3) Mexico
#   4) Colombia
#
# Download the full tropical belt once; crop in memory for regional maps.
# GFS latitude/longitude coordinates are CELL CENTERS.
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
import requests

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

from eccodes import (
    codes_grib_new_from_file, codes_get, codes_get_array,
    codes_get_values, codes_release,
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "forecast_tropical"
JSON = OUTPUT / "gfs_tropical_precip_forecast_latest.json"

parser = argparse.ArgumentParser(
    description="Generate multi-region tropical GFS precipitation forecast frames."
)
parser.add_argument(
    "--output-dir", type=Path, default=None,
    help="Alternative destination, e.g. sandbox/gfs_tropical_test"
)
parser.add_argument(
    "--force", action="store_true",
    help="Generate again even if this GFS cycle is already published"
)
args = parser.parse_args()

if args.output_dir is not None:
    OUTPUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    JSON = OUTPUT / "gfs_tropical_precip_forecast_latest.json"

OUTPUT.mkdir(parents=True, exist_ok=True)

cartopy.config["data_dir"] = ROOT / ".cache" / "cartopy"
Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

NOMADS_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
FORECAST_HOURS = list(range(6, 49, 6))
LAT_MIN, LAT_MAX = -35, 35

# Same regions, extents, and figure sizes used in the IMERG product.
MAPS = [
    ("global_tropics", "Global Tropical Forecast Precipitation",
     [-180, 180, -35, 35], (18, 6.4), 60, 15),
    ("americas", "Tropical Forecast Precipitation over the Americas",
     [-120, -30, -35, 35], (12, 8), 20, 10),
    ("mexico", "6-hour Forecast Precipitation over Mexico",
     [-118, -86, 14, 33], (12, 7), 5, 5),
    ("colombia", "6-hour Forecast Precipitation over Colombia",
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
# 4. FIND THE MOST RECENT COMPLETE GFS CYCLE
# =============================================================================

now = datetime.now(timezone.utc)
latest_hour = (now.hour // 6) * 6
latest_cycle = now.replace(hour=latest_hour, minute=0, second=0, microsecond=0)
cycles = [latest_cycle - timedelta(hours=6 * i) for i in range(6)]

session = requests.Session()
session.headers.update({"User-Agent": "TropicalWeatherClimateObservatory/0.1 (research)"})

selected_cycle = None
forecast_fields = None
lat = None
lon = None
source_urls = []

print("TWCO | GFS Tropical Forecast Precipitation")
print("Script version: regional crop + pixel edges + lossless WebP")
print(f"Script path: {Path(__file__).resolve()}")
print(f"Output directory: {OUTPUT}")
print("Independent 6-hour accumulations, +0 to +48 h")
print("Download domain: 35S–35N, all longitudes")

with tempfile.TemporaryDirectory(prefix="twco_gfs_tropical_") as temp_dir:
    temp_dir = Path(temp_dir)

    for cycle in cycles:
        print(f"\nChecking GFS cycle {cycle:%Y-%m-%d %H}:00 UTC...")
        urls = []
        downloaded = {}
        previous_lat = None
        previous_lon = None

        try:
            # Check f048 first to avoid downloading an incomplete fresh cycle.
            leads = [48] + [hour for hour in FORECAST_HOURS if hour != 48]

            for lead in leads:
                grib_file = temp_dir / f"gfs_f{lead:03d}.grib2"
                params = {
                    "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}",
                    "lev_surface": "on",
                    "var_APCP": "on",
                    "subregion": "",
                    "leftlon": 0,
                    "rightlon": 359.75,
                    "toplat": LAT_MAX,
                    "bottomlat": LAT_MIN,
                    "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
                }

                response = session.get(NOMADS_URL, params=params, timeout=(20, 180))
                response.raise_for_status()
                grib_file.write_bytes(response.content)
                urls.append(response.url)

                if grib_file.stat().st_size < 1000 or response.content[:4] != b"GRIB":
                    raise ValueError(f"f{lead:03d}: NOMADS did not return a valid GRIB2 file")

                matches = []
                intervals_found = []

                with grib_file.open("rb") as handle:
                    while True:
                        gid = codes_grib_new_from_file(handle)
                        if gid is None:
                            break
                        try:
                            short_name = str(codes_get(gid, "shortName")).lower()
                            name = str(codes_get(gid, "name")).lower()
                            step_type = str(codes_get(gid, "stepType"))
                            level = str(codes_get(gid, "typeOfLevel"))

                            if not (short_name in ("tp", "apcp") or name == "total precipitation"):
                                continue
                            if step_type != "accum" or level != "surface":
                                continue

                            start_step = int(codes_get(gid, "startStep"))
                            end_step = int(codes_get(gid, "endStep"))
                            step_units = str(codes_get(gid, "stepUnits")).lower()
                            intervals_found.append((start_step, end_step, step_units))

                            if step_units not in ("h", "1", "hour", "hours"):
                                raise ValueError(f"Unexpected GRIB time units: {step_units}")
                            if (start_step, end_step) != (lead - 6, lead):
                                continue

                            date = int(codes_get(gid, "dataDate"))
                            time = int(codes_get(gid, "dataTime"))
                            if date != int(cycle.strftime("%Y%m%d")) or time != int(cycle.strftime("%H%M")):
                                raise ValueError(f"f{lead:03d}: wrong GFS initialization time")

                            units = str(codes_get(gid, "units"))
                            if units not in ("kg m**-2", "kg m-2", "kg/m^2", "mm"):
                                raise ValueError(f"Unexpected APCP units: {units}")

                            values = np.asarray(codes_get_values(gid), dtype=float)
                            lats = np.asarray(codes_get_array(gid, "latitudes"), dtype=float)
                            lons = np.asarray(codes_get_array(gid, "longitudes"), dtype=float)

                            # Convert GFS longitudes from 0..360 to -180..180.
                            lons = np.where(lons >= 180, lons - 360, lons)

                            lat_1d = np.unique(np.round(lats, 5))
                            lon_1d = np.unique(np.round(lons, 5))

                            if len(lat_1d) * len(lon_1d) != len(values):
                                raise ValueError("APCP data are not on a regular grid")

                            field = np.full((len(lat_1d), len(lon_1d)), np.nan)
                            iy = np.searchsorted(lat_1d, np.round(lats, 5))
                            ix = np.searchsorted(lon_1d, np.round(lons, 5))
                            field[iy, ix] = values

                            if not np.isfinite(field).all():
                                raise ValueError("APCP contains missing or invalid grid cells")
                            if np.min(field) < -0.1 or np.max(field) > 1500:
                                raise ValueError("Unreasonable APCP values")

                            # Require complete tropical latitude coverage.
                            if lat_1d.min() > LAT_MIN or lat_1d.max() < LAT_MAX:
                                raise ValueError("Downloaded field does not cover the tropical latitude band")
                            # Require complete longitude coverage after the conversion.
                            if len(lon_1d) != 1440 or not np.allclose(
                                lon_1d, np.arange(-180, 180, 0.25), atol=1e-5
                            ):
                                raise ValueError("Downloaded field does not contain the full 0.25-degree longitude grid")
                            if len(lat_1d) != 281 or not np.allclose(
                                lat_1d, np.arange(-35, 35.25, 0.25), atol=1e-5
                            ):
                                raise ValueError("Downloaded field does not contain the full tropical latitude grid")

                            matches.append((lat_1d, lon_1d, np.maximum(field, 0.0)))

                        finally:
                            codes_release(gid)

                grib_file.unlink(missing_ok=True)

                if not matches:
                    raise ValueError(
                        f"f{lead:03d}: no exact {lead-6}-{lead} h APCP record; "
                        f"available intervals: {intervals_found}"
                    )

                if any(not np.array_equal(m[2], matches[0][2]) for m in matches[1:]):
                    raise ValueError(f"f{lead:03d}: conflicting APCP records")

                current_lat, current_lon, field = matches[0]

                if previous_lat is not None and (
                    not np.array_equal(current_lat, previous_lat) or
                    not np.array_equal(current_lon, previous_lon)
                ):
                    raise ValueError("GFS grid changed between forecast frames")

                previous_lat = current_lat
                previous_lon = current_lon
                downloaded[lead] = field
                print(f"  f{lead:03d} | {lead-6:02d}-{lead:02d} h | max {field.max():.1f} mm")

            lat = previous_lat
            lon = previous_lon
            forecast_fields = [downloaded[h] for h in FORECAST_HOURS]
            selected_cycle = cycle
            source_urls = urls
            break

        except (requests.RequestException, OSError, ValueError) as error:
            print(f"  Cycle incomplete or incompatible: {str(error)[:240]}")
            for old_grib in temp_dir.glob("*.grib2"):
                old_grib.unlink(missing_ok=True)

    session.close()

if selected_cycle is None:
    raise RuntimeError("No complete GFS cycle with exact 6-hour APCP windows was available")

print(f"\nSelected GFS cycle: {selected_cycle:%Y-%m-%d %H}:00 UTC")

# Never overwrite a newer published cycle unless --force was requested.
if JSON.exists() and not args.force:
    try:
        previous = json.loads(JSON.read_text(encoding="utf-8"))
        previous_cycle = datetime.fromisoformat(previous["gfs_cycle_utc"])
        if previous_cycle >= selected_cycle:
            print("Already published this GFS cycle or a newer one. No change.")
            raise SystemExit(0)
    except (KeyError, ValueError, json.JSONDecodeError):
        print("Existing metadata could not be read; regenerating.")

# =============================================================================
# 5. GRID GEOMETRY: GFS CELL CENTERS -> PIXEL EDGES
# =============================================================================
# The GFS GRIB coordinates identify each cell CENTER.
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
    raise RuntimeError("The GFS cell edges do not match the grid centers")

# =============================================================================
# 6. RENDER AND PUBLISH
# =============================================================================

plt.rcParams["font.family"] = "DejaVu Sans"
expected_frames = len(MAPS) * len(FORECAST_HOURS)

with tempfile.TemporaryDirectory(prefix=".twco_publish_", dir=OUTPUT) as publish_tmp:
    publish_tmp = Path(publish_tmp)

    metadata = {
        "product": "TWCO-GFS-TROPICAL",
        "title": "GFS Tropical Forecast Precipitation",
        "source": "NOAA/NCEP GFS 0.25°",
        "gfs_cycle_utc": selected_cycle.isoformat(),
        "forecast_hours": FORECAST_HOURS,
        "interval_hours": 6,
        "domain": "35S-35N, all longitudes",
        "units": "mm / 6 h",
        "image_format": "webp",
        "compression": "lossless",
        "grid_resolution_degrees": 0.25,
        "pixel_coordinates": "GFS centers converted to cell edges for pcolormesh(shading=flat)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regions": [],
        "source_urls": source_urls,
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

        # Select the GFS subregion once. No need to send the whole global
        # matrix to Cartopy for the Americas, Mexico or Colombia.
        # Include a one-cell margin at regional borders.
        if key == "global_tropics":
            iy0, iy1 = 0, lat.size
            ix0, ix1 = 0, lon.size
        else:
            pad = 0.25  # one GFS grid cell
            iy = np.flatnonzero((lat >= extent[2] - pad) & (lat <= extent[3] + pad))
            ix = np.flatnonzero((lon >= extent[0] - pad) & (lon <= extent[1] + pad))
            if iy.size < 2 or ix.size < 2:
                raise RuntimeError(f"Insufficient GFS grid cells for {key}")
            iy0, iy1 = int(iy[0]), int(iy[-1]) + 1
            ix0, ix1 = int(ix[0]), int(ix[-1]) + 1

        region_lat_edges = lat_edges[iy0:iy1 + 1]
        region_lon_edges = lon_edges[ix0:ix1 + 1]
        # A field with (ny, nx) cells needs (ny+1, nx+1) pixel edges.
        if region_lat_edges.size != (iy1 - iy0 + 1) or region_lon_edges.size != (ix1 - ix0 + 1):
            raise RuntimeError(f"Pixel edges do not match the {key} crop")
        print(f"  GFS grid for map: {iy1 - iy0} x {ix1 - ix0} cells")

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
                "Source: NOAA/NCEP GFS 0.25°  |  Surface APCP  |  Exact 6-hour interval verified",
                fontsize=9, color="#64748b"
            )
            fig.text(
                0.945, 0.065,
                "ajaramillomoreno.github.io",
                fontsize=9, color="#64748b", ha="right"
            )

            filename = f"gfs_precip_f{lead:03d}.webp"
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
            source = publish_tmp / "frames" / key / f"gfs_precip_f{lead:03d}.webp"
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
            filename = f"gfs_precip_f{lead:03d}.webp"
            source = src_dir / filename
            target = dst_dir / filename

            if not source.is_file():
                raise RuntimeError(f"Missing forecast frame: {source}")

            os.replace(source, target)
            print(f"Saved: {target}")

    os.replace(publish_tmp / JSON.name, JSON)
    print(f"Metadata saved: {JSON}")

print(f"\nDone. {expected_frames} lossless WebP frames and JSON manifest generated.")
