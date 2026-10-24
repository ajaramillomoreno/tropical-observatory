
# =============================================================================
# Tropical Weather & Climate Observatory (TWCO)
# Product: Daily OLR and OLR Anomaly
# Source: NOAA CPC Blended OLR (1-degree daily binary files)
# OLR anomaly climatology: 1991-2020
# Regions: Global Tropics and Tropical Pacific
#
# Author: Alejandro Jaramillo Moreno
# Website: https://ajaramillomoreno.github.io
# =============================================================================

from pathlib import Path
from datetime import datetime, timezone
import argparse
import io
import json
import os
import re
import tempfile

import requests
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from PIL import Image


# =============================================================================
# 1. SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "climate_state"

parser = argparse.ArgumentParser(description="Generate daily tropical OLR maps.")
parser.add_argument(
    "--output-dir", type=Path, default=None,
    help="Alternative output directory, e.g. sandbox/olr_test"
)
parser.add_argument(
    "--force", action="store_true",
    help="Regenerate even if the latest date is already published"
)
args = parser.parse_args()

if args.output_dir is not None:
    OUTPUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

OUTPUT.mkdir(parents=True, exist_ok=True)
JSON = OUTPUT / "tropical_olr_latest.json"

cartopy.config["data_dir"] = ROOT / ".cache" / "cartopy"
Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

BASE = "https://ftp.cpc.ncep.noaa.gov/precip/CBO_V1/1deg-DLY"

MAPS = [
    ("global_tropics", "Global Tropics", (-180, 180), (18, 6.4), 60),
    ("tropical_pacific", "Tropical Pacific", (-80, 110), (15, 8.5), 30),
]

# Discrete color scales (both colorbars use extend='both').
olr_levels = np.arange(100, 341, 10)
anom_levels = np.arange(-80, 81, 5)

olr_cmap = ListedColormap(
    plt.get_cmap("turbo")(np.linspace(0, 1, len(olr_levels) - 1))
)
anom_cmap = ListedColormap(
    plt.get_cmap("RdBu_r")(np.linspace(0, 1, len(anom_levels) - 1))
)

olr_cmap.set_under("#30123b")
olr_cmap.set_over("#800000")
anom_cmap.set_under("#053061")
anom_cmap.set_over("#67001f")
olr_cmap.set_bad("#f2f4f6")
anom_cmap.set_bad("#f2f4f6")

PRODUCTS = [
    ("olr", "Outgoing Longwave Radiation",
     olr_cmap, BoundaryNorm(olr_levels, olr_cmap.N),
     np.arange(100, 341, 40), "OLR (W m$^{-2}$)"),
    ("olr_anomaly", "Outgoing Longwave Radiation Anomaly",
     anom_cmap, BoundaryNorm(anom_levels, anom_cmap.N),
     np.arange(-80, 81, 20), "OLR anomaly (W m$^{-2}$)"),
]

expected_images = [
    OUTPUT / f"{product}_{region}_latest.webp"
    for product, *_ in PRODUCTS
    for region, *_ in MAPS
]
cache_ready = (
    JSON.is_file()
    and all(p.is_file() and p.stat().st_size > 0 for p in expected_images)
)


# =============================================================================
# 2. FIND THE MOST RECENT CPC DAILY FILE
# =============================================================================

print("TWCO | Daily OLR and OLR Anomaly (native 1-degree grid)")
print(f"Output directory: {OUTPUT}")
print("Source: NOAA Climate Prediction Center")

today = datetime.now(timezone.utc).date()
latest_date = None
source_url = None

try:
    # At New Year, the previous year's directory is a useful fallback.
    for year in [today.year, today.year - 1]:
        index_url = f"{BASE}/{year}/"
        response = requests.get(index_url, timeout=30)

        if response.status_code == 404:
            continue
        response.raise_for_status()

        dates = re.findall(r"CBO_V1_1deg-DLY\.lnx\.(\d{8})", response.text)
        dates = sorted(set(d for d in dates if d <= today.strftime("%Y%m%d")))

        if dates:
            ymd = dates[-1]
            latest_date = datetime.strptime(ymd, "%Y%m%d").date()
            source_url = f"{BASE}/{year}/CBO_V1_1deg-DLY.lnx.{ymd}"
            break

except requests.RequestException as error:
    if cache_ready:
        print(f"NOAA unavailable ({error}). Keeping existing figures.")
        raise SystemExit(0)
    raise

if latest_date is None:
    if cache_ready:
        print("No recent NOAA file found. Keeping existing figures.")
        raise SystemExit(0)
    raise RuntimeError("Could not locate CPC daily OLR files.")

print(f"Latest available date: {latest_date}")

if cache_ready and not args.force:
    try:
        previous = json.loads(JSON.read_text(encoding="utf-8"))
        if previous["date"] >= latest_date.isoformat():
            print("OLR products are already up to date.")
            raise SystemExit(0)
    except (KeyError, ValueError, json.JSONDecodeError):
        print("Previous metadata invalid. Regenerating figures.")


# =============================================================================
# 3. DOWNLOAD AND READ THE TWO FIELDS
# =============================================================================

# NOAA GrADS control file specifies:
#   little-endian float32; 360 longitudes, 180 latitudes;
#   first field = OLR, second field = OLR anomaly.
# No spatial subsampling or calculation of anomalies is needed.

print(f"Downloading: {source_url}")

try:
    response = requests.get(source_url, timeout=120)
    response.raise_for_status()

    expected_bytes = 2 * 180 * 360 * 4
    if len(response.content) != expected_bytes:
        raise ValueError(
            f"Unexpected CPC file size: {len(response.content)} bytes "
            f"(expected {expected_bytes})."
        )

    raw = np.frombuffer(response.content, dtype="<f4").reshape(2, 180, 360)
    olr = np.where(raw[0] <= -900, np.nan, raw[0]).copy()
    anomaly = np.where(raw[1] <= -900, np.nan, raw[1]).copy()

    if not np.isfinite(olr).any() or not np.isfinite(anomaly).any():
        raise ValueError("CPC file contains no valid OLR data.")

    # Guard against wrong decoding: typical global OLR is around 200-270 W/m2.
    if not 100 < np.nanmedian(olr) < 350:
        raise ValueError("OLR values are implausible; check the binary format.")

except (requests.RequestException, ValueError) as error:
    if cache_ready:
        print(f"Unable to read new CPC data ({error}). Keeping existing figures.")
        raise SystemExit(0)
    raise

print("OLR and anomaly fields loaded: 180 x 360 (1 degree)")

# CPC grid: longitude 0.5-359.5 E, latitude -89.5-89.5 N.
lon = np.arange(0.5, 360, 1)
lat = np.arange(-89.5, 90, 1)

# Limit the data to 35S-35N (70 grid rows).
tropical = (lat >= -35) & (lat <= 35)
lat = lat[tropical]
fields = {"olr": olr[tropical, :], "olr_anomaly": anomaly[tropical, :]}


# =============================================================================
# 4. GENERATE MAPS AND SAVE OUTPUTS
# =============================================================================

plt.rcParams["font.family"] = "DejaVu Sans"
date_label = latest_date.strftime("%d %b %Y")
projection = ccrs.PlateCarree(central_longitude=180)

# Use the same disk as OUTPUT for temporary files (external-SSD safe).
with tempfile.TemporaryDirectory(prefix=".twco_olr_", dir=OUTPUT) as tmp:
    temporary = Path(tmp)

    for product, title, cmap, norm, ticks, unit in PRODUCTS:
        for region, region_title, xlim, figsize, lon_step in MAPS:
            print(f"Rendering {product} | {region}")

            # Longitudes are converted to the 180-centered map coordinates.
            if region == "global_tropics":
                region_lon = lon - 180
                values = fields[product]
            else:
                pacific = (lon >= 100) & (lon <= 290)
                region_lon = lon[pacific] - 180
                values = fields[product][:, pacific]

            extent = [region_lon[0] - 0.5, region_lon[-1] + 0.5,
                      lat[0] - 0.5, lat[-1] + 0.5]

            fig = plt.figure(figsize=figsize, facecolor="white")
            ax = fig.add_axes([0.055, 0.27, 0.89, 0.49], projection=projection)
            ax.set_extent([*xlim, -35, 35], crs=projection)

            mesh = ax.imshow(
                np.ma.masked_invalid(values),
                origin="lower", extent=extent, transform=projection,
                interpolation="nearest", cmap=cmap, norm=norm,
            )

            # OLR is defined over land and ocean.
            ax.coastlines(resolution="110m", linewidth=0.7, color="#273847")
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, lon_step), crs=projection)
            ax.set_yticks(np.arange(-30, 31, 15), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.gridlines(linewidth=0.4, linestyle=":", alpha=0.30)
            ax.tick_params(labelsize=9)

            fig.text(0.055, 0.95, "TROPICAL WEATHER & CLIMATE OBSERVATORY",
                     fontsize=11, weight="bold", color="#1c405b")
            fig.text(0.055, 0.885, f"{title} — {region_title}",
                     fontsize=18, weight="bold", color="#152a3c")

            subtitle = f"Valid: {date_label}  |  NOAA CPC Blended OLR 1°"
            if product == "olr_anomaly":
                subtitle += "  |  Climatology: 1991–2020"
            fig.text(0.055, 0.825, subtitle, fontsize=10, color="#475569")

            # Center the horizontal colorbar below the actual map width.
            fig.canvas.draw()
            map_box = ax.get_position()
            cb_width = map_box.width * 0.82
            cb_left = map_box.x0 + (map_box.width - cb_width) / 2
            cax = fig.add_axes([cb_left, 0.17, cb_width, 0.025])
            cb = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                              extend="both", ticks=ticks, spacing="uniform")
            cb.set_label(unit, fontsize=11)
            cb.ax.tick_params(labelsize=9)

            fig.text(0.055, 0.065,
                     "Source: NOAA CPC Blended OLR | Daily 1° satellite analysis",
                     fontsize=9, color="#64748b")
            fig.text(0.945, 0.065, "ajaramillomoreno.github.io",
                     fontsize=9, ha="right", color="#64748b")

            destination = temporary / f"{product}_{region}_latest.webp"
            buffer = io.BytesIO()
            fig.savefig(buffer, dpi=200, format="png", facecolor="white")
            plt.close(fig)
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.save(destination, "WEBP", lossless=True, method=6)
            buffer.close()
            print(f"Generated: {destination.name}")

    metadata = {
        "product": "TWCO-DAILY-OLR",
        "source": "NOAA CPC Blended OLR V1, native 1-degree daily binary",
        "date": latest_date.isoformat(),
        "climatology_olr_anomaly": "1991-2020",
        "data_source_url": source_url,
        "image_format": "webp",
        "compression": "lossless",
        "grid_resolution_degrees": 1.0,
        "spatial_subsampling": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regions": {},
    }

    for region, region_title, *_ in MAPS:
        metadata["regions"][region] = {
            "title": region_title,
            "olr_image": f"olr_{region}_latest.webp",
            "olr_anomaly_image": f"olr_anomaly_{region}_latest.webp",
        }

    (temporary / JSON.name).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Verify the full set before replacing any previously published map.
    for destination in expected_images:
        source = temporary / destination.name
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure: {source}")

    for destination in expected_images:
        os.replace(temporary / destination.name, destination)
        print(f"Saved: {destination}")

    os.replace(temporary / JSON.name, JSON)
    print(f"Saved metadata: {JSON}")

print("\nDone. Four daily OLR maps and JSON generated.")
