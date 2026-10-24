# =============================================================================
# Tropical Weather & Climate Observatory (TWCO)
# Daily Sea Surface Temperature (SST) and SST Anomaly
# NOAA Coral Reef Watch CoralTemp v3.1 | Native 0.05-degree grid (~5 km)
# SST anomaly reference period: 1991-2020
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

import requests
import numpy as np
import xarray as xr

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

parser = argparse.ArgumentParser(description="Generate daily tropical SST maps.")
parser.add_argument(
    "--output-dir", type=Path, default=None,
    help="Alternative directory, e.g. sandbox/sst_test"
)
parser.add_argument(
    "--force", action="store_true",
    help="Regenerate even if the latest available date has already been published"
)
args = parser.parse_args()

if args.output_dir is not None:
    OUTPUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

OUTPUT.mkdir(parents=True, exist_ok=True)
JSON = OUTPUT / "tropical_sst_latest.json"

cartopy.config["data_dir"] = ROOT / ".cache" / "cartopy"
Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

SST_BASE = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/"
    "v3.1_op/nc/v1.0/daily/sst"
)
ANOM_BASE = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/"
    "v3.1-clim19912020-v1/nc/v1.0/daily/ssta"
)

# Region, title, extent in PlateCarree(180), figure size, longitude tick spacing.
MAPS = [
    ("global_tropics", "Global Tropics", (-180, 180), (18, 6.4), 60),
    ("tropical_pacific", "Tropical Pacific", (-80, 110), (15, 8.5), 30),
]

# Discrete color scales: SST every 1 deg C; SSTA every 0.1 deg C.
sst_levels = np.arange(0, 36, 1)
anom_levels = np.round(np.linspace(-4, 4, 81), 1)

sst_cmap = ListedColormap(plt.get_cmap("turbo")(np.linspace(0, 1, len(sst_levels) - 1)))
anom_cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0, 1, len(anom_levels) - 1)))

sst_cmap.set_under("#30123b")
sst_cmap.set_over("#800000")
anom_cmap.set_under("#053061")
anom_cmap.set_over("#67001f")

sst_norm = BoundaryNorm(sst_levels, sst_cmap.N)
anom_norm = BoundaryNorm(anom_levels, anom_cmap.N)

PRODUCTS = [
    ("sst", "Sea Surface Temperature", sst_cmap, sst_norm,
     np.arange(0, 36, 5), "SST (°C)"),
    ("sst_anomaly", "Sea Surface Temperature Anomaly", anom_cmap, anom_norm,
     np.arange(-4, 5, 1), "SST anomaly (°C)"),
]

expected_images = [
    OUTPUT / f"{product}_{region}_latest.webp"
    for product, *_ in PRODUCTS
    for region, *_ in MAPS
]


# =============================================================================
# 2. FIND THE LATEST COMMON DAILY SST AND SST ANOMALY
# =============================================================================

print("TWCO | Daily SST and SST Anomaly (original 0.05° grid)")
print(f"Output directory: {OUTPUT}")
print("Checking NOAA Coral Reef Watch files...")

latest_date = None
sst_url = None
anom_url = None

today = datetime.now(timezone.utc).date()

for days_back in range(11):
    date = today - timedelta(days=days_back)
    ymd = date.strftime("%Y%m%d")
    year = date.strftime("%Y")

    candidate_sst = f"{SST_BASE}/{year}/coraltemp_v3.1_{ymd}.nc"
    candidate_anom = (
        f"{ANOM_BASE}/{year}/ct5km_ssta_v3.1-clim19912020-v1_{ymd}.nc"
    )

    try:
        sst_ok = requests.head(candidate_sst, timeout=15).status_code == 200
        anom_ok = requests.head(candidate_anom, timeout=15).status_code == 200
    except requests.RequestException:
        continue

    if sst_ok and anom_ok:
        latest_date = date
        sst_url = candidate_sst
        anom_url = candidate_anom
        break

if latest_date is None:
    if JSON.exists() and all(p.exists() and p.stat().st_size > 0 for p in expected_images):
        print("No recent NOAA files found; keeping the existing valid figures.")
        raise SystemExit(0)
    raise RuntimeError("No matching recent SST and SST anomaly files found at NOAA.")

print(f"Latest common observation date: {latest_date}")

if JSON.exists() and not args.force:
    try:
        previous = json.loads(JSON.read_text(encoding="utf-8"))
        if (previous["date"] >= latest_date.isoformat()
                and all(p.exists() and p.stat().st_size > 0 for p in expected_images)):
            print("All four figures are already up to date.")
            raise SystemExit(0)
    except (KeyError, ValueError, json.JSONDecodeError):
        print("Metadata missing or invalid. Regenerating figures.")


# =============================================================================
# 3. DOWNLOAD AND READ THE TWO DAILY FIELDS
# =============================================================================

# The temporary directory is inside OUTPUT (also works with an external SSD).
with tempfile.TemporaryDirectory(prefix=".twco_sst_", dir=OUTPUT) as tmp:
    temporary = Path(tmp)
    fields = {}

    for key, url in [("sst", sst_url), ("sst_anomaly", anom_url)]:
        print(f"\nDownloading {key}: {url}")
        filename = temporary / f"{key}.nc"

        with requests.get(url, timeout=180, stream=True) as response:
            response.raise_for_status()
            with filename.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        target.write(chunk)

        with xr.open_dataset(filename) as ds:
            variable = (
                "analysed_sst" if key == "sst"
                else "sea_surface_temperature_anomaly"
            )
            field = ds[variable].squeeze(drop=True)
            field = field.rename({
                name: "lat" for name in field.dims if name == "latitude"
            })
            field = field.rename({
                name: "lon" for name in field.dims if name == "longitude"
            })

            # Only select the tropical belt: no subsampling or interpolation.
            lat_slice = slice(-35, 35) if float(field.lat[0]) < float(field.lat[-1]) else slice(35, -35)
            field = field.sel(lat=lat_slice)
            field = field.assign_coords(lon=field.lon % 360).sortby("lon")
            field = field.sortby("lat").load()

            fields[key] = field
            print(f"  {variable}: {field.sizes['lat']} × {field.sizes['lon']} grid points")

    # =========================================================================
    # 4. GENERATE FOUR MAPS
    # =========================================================================

    plt.rcParams["font.family"] = "DejaVu Sans"
    date_label = latest_date.strftime("%d %b %Y")
    projection = ccrs.PlateCarree(central_longitude=180)

    for product, title, cmap, norm, ticks, unit in PRODUCTS:
        field = fields[product]

        for region, region_title, xlim, figsize, lon_step in MAPS:
            print(f"\nRendering {product} | {region}")

            regional = field if region == "global_tropics" else field.sel(lon=slice(100, 290))
            lon = regional.lon.values - 180.0
            lat = regional.lat.values
            dx = float(lon[1] - lon[0])
            dy = float(lat[1] - lat[0])

            image_extent = [
                lon[0] - dx / 2, lon[-1] + dx / 2,
                lat[0] - dy / 2, lat[-1] + dy / 2,
            ]

            fig = plt.figure(figsize=figsize, facecolor="white")

            # Keep the entire map below the date/subtitle on both regions.
            # Pacific figure is taller so the map can use more horizontal space.
            ax = fig.add_axes([0.055, 0.27, 0.89, 0.49], projection=projection)
            ax.set_extent([xlim[0], xlim[1], -35, 35], crs=projection)

            mesh = ax.imshow(
                np.ma.masked_invalid(regional.values),
                origin="lower", extent=image_extent,
                transform=projection, interpolation="nearest",
                cmap=cmap, norm=norm,
            )

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
            subtitle = f"Valid: {date_label}  |  NOAA Coral Reef Watch CoralTemp v3.1"
            if product == "sst_anomaly":
                subtitle += "  |  Climatology: 1991–2020"
            fig.text(0.055, 0.825, subtitle, fontsize=10, color="#475569")

            # Draw map first so the colorbar can be centered under its actual width.
            fig.canvas.draw()
            map_box = ax.get_position()
            cb_width = map_box.width * 0.82
            cb_left = map_box.x0 + (map_box.width - cb_width) / 2
            cax = fig.add_axes([cb_left, 0.17, cb_width, 0.025])

            cb = fig.colorbar(
                mesh, cax=cax, orientation="horizontal",
                extend="both", ticks=ticks, spacing="uniform"
            )
            cb.set_label(unit, fontsize=11)
            cb.ax.tick_params(labelsize=9)

            fig.text(0.055, 0.065,
                     "Source: NOAA Coral Reef Watch | Daily 5-km satellite SST",
                     fontsize=9, color="#64748b")
            fig.text(0.945, 0.065, "ajaramillomoreno.github.io",
                     fontsize=9, ha="right", color="#64748b")

            # Enough pixels to show all native grid cells, without downsampling
            # the data before plotting. Export to lossless WebP.
            dpi = 500 if region == "global_tropics" else 420
            destination = temporary / f"{product}_{region}_latest.webp"
            buffer = io.BytesIO()
            fig.savefig(buffer, dpi=dpi, format="png", facecolor="white")
            plt.close(fig)
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.save(destination, "WEBP", lossless=True, method=6)
            buffer.close()
            print(f"Generated: {destination.name}")

    # =========================================================================
    # 5. METADATA AND PUBLICATION
    # =========================================================================

    metadata = {
        "product": "TWCO-DAILY-SST",
        "source": "NOAA Coral Reef Watch CoralTemp v3.1",
        "date": latest_date.isoformat(),
        "climatology_sst_anomaly": "1991-2020",
        "sst_source_url": sst_url,
        "sst_anomaly_source_url": anom_url,
        "image_format": "webp",
        "compression": "lossless",
        "grid_resolution_degrees": 0.05,
        "spatial_subsampling": False,
        "regions": {},
    }

    for region, region_title, *_ in MAPS:
        metadata["regions"][region] = {
            "title": region_title,
            "sst_image": f"sst_{region}_latest.webp",
            "sst_anomaly_image": f"sst_anomaly_{region}_latest.webp",
        }

    (temporary / JSON.name).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Verify all figures exist before replacing previous outputs.
    for destination in expected_images:
        source = temporary / destination.name
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure: {source}")

    for destination in expected_images:
        os.replace(temporary / destination.name, destination)
        print(f"Saved: {destination}")

    os.replace(temporary / JSON.name, JSON)
    print(f"Saved metadata: {JSON}")

print("\nDone. Four daily SST figures and JSON generated.")
