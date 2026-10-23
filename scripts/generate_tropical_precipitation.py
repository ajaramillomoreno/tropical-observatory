# =============================================================================
# Tropical Weather & Climate Observatory (TWCO)
# Product: Global Tropical Precipitation | 24-hour Accumulation
#
# Author      : Alejandro Jaramillo Moreno
# Email       : ajaramillo@atmosfera.unam.mx
# Website     : https://ajaramillomoreno.github.io
# Affiliation : Departamento de Cambio Climático
#               Instituto de Ciencias de la Atmósfera y Cambio Climático
#               Universidad Nacional Autónoma de México (UNAM)
# Location    : Mexico City, Mexico
# =============================================================================

import argparse
import io
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import earthaccess
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save figures without a display (GitHub Actions)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


# =============================================================================
# 1. SETTINGS
# =============================================================================

PRODUCT = "GPM_3IMERGHHE"   # NASA IMERG Early Run, half-hourly
VERSION = "07"              # NASA collection version; files may be V07B/V07C
LAT_MIN, LAT_MAX = -35, 35
N_FILES = 48                # 48 x 30 min = 24 hours

# Repository root
ROOT = Path(__file__).resolve().parents[1]

# Run normally to publish to outputs/, or specify a sandbox test directory.
parser = argparse.ArgumentParser(description="Generate IMERG 24-hour precipitation maps")
parser.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory, e.g. sandbox/imerg_test")
args = parser.parse_args()

OUTPUT = ROOT / "outputs" / "precipitation"
if args.output_dir is not None:
    OUTPUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)

JSON = OUTPUT / "tropical_precip_24h_latest.json"


# Each map uses the same accumulated IMERG field and color scale.
MAPS = [
    ("Global Tropical Precipitation", [-180, 180, -35, 35],
     "tropical_precip_24h_latest.webp", (18, 6.4), 60, 15),
    ("Tropical Precipitation over the Americas", [-120, -30, -35, 35],
     "americas_precip_24h_latest.webp", (12, 8), 20, 10),
    ("24-hour Precipitation over Mexico", [-118, -86, 14, 33],
     "mexico_precip_24h_latest.webp", (12, 7), 5, 5),
    ("24-hour Precipitation over Colombia", [-84, -66, -6, 15],
     "colombia_precip_24h_latest.webp", (9, 9), 4, 5),
]


# =============================================================================
# 2. NASA EARTHDATA LOGIN
# =============================================================================

print("Logging into NASA Earthdata...")

if os.getenv("GITHUB_ACTIONS") == "true":
    if not (os.getenv("EARTHDATA_TOKEN") or
            (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"))):
        raise RuntimeError("Configure your Earthdata credentials as GitHub Actions Secrets.")
    auth = earthaccess.login(strategy="environment")
else:
    auth = earthaccess.login()  # Interactive or stored login on your computer

if not auth.authenticated:
    raise RuntimeError("NASA Earthdata authentication failed.")


# =============================================================================
# 3. FIND THE LATEST COMPLETE 24-HOUR WINDOW
# =============================================================================

print("Searching for IMERG Early Run data...")

now = datetime.now(timezone.utc)
granules = earthaccess.search_data(
    short_name=PRODUCT,
    version=VERSION,
    temporal=(now - timedelta(hours=72), now),
)

# Index granules by the beginning of each half-hourly observation.
by_time = {}
for g in granules:
    stamp = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    time = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    by_time[time] = g

step = timedelta(minutes=30)
selected_times = None

# Search backwards until there are 48 CONSECUTIVE files.
for last in sorted(by_time, reverse=True):
    times = [last - step * (N_FILES - 1 - i) for i in range(N_FILES)]
    if all(t in by_time for t in times):
        selected_times = times
        break

if selected_times is None:
    raise RuntimeError("No 24-hour period with 48 consecutive IMERG files was found.")

start = selected_times[0]
end = selected_times[-1] + step  # Exclusive: exactly 24 hours

if now - end > timedelta(hours=36):
    raise RuntimeError("The latest complete window is too old; keeping the previous map.")

selected = [by_time[t] for t in selected_times]
print(f"Selected: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")


# =============================================================================
# 4. DOWNLOAD TEMPORARY FILES, PROCESS, AND PUBLISH
# =============================================================================

# TemporaryDirectory removes all downloads, including if an error occurs.
with tempfile.TemporaryDirectory(prefix="twco_imerg_") as temporary:
    temporary = Path(temporary)
    print(f"Downloading {N_FILES} IMERG files...")
    downloads = earthaccess.download(selected, local_path=str(temporary), threads=6)

    # Match each download to its actual start time; download order is not assumed.
    files = {}
    file_versions = set()
    for item in downloads:
        path = Path(item)
        match = re.search(r"(\d{8})-S(\d{6})-E", path.name)
        version_match = re.search(r"\.(V\d+[A-Z])\.HDF5$", path.name, re.I)
        if not match or not version_match or not path.is_file():
            raise RuntimeError(f"Unexpected or missing downloaded file: {path}")
        time = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        time = time.replace(tzinfo=timezone.utc)
        if time in files:
            raise RuntimeError(f"Duplicate IMERG file for {time}")
        files[time] = path
        file_versions.add(version_match.group(1).upper())

    if set(files) != set(selected_times):
        raise RuntimeError("Missing or unexpected IMERG intervals; map not updated.")
    if len(file_versions) != 1:
        raise RuntimeError(f"Different IMERG file versions in one window: {file_versions}")
    file_version = file_versions.pop()

    print(f"Processing {N_FILES} files ({file_version})...")
    accumulated = None
    valid_all = None

    for i, time in enumerate(selected_times, start=1):
        with h5py.File(files[time], "r") as f:
            grid = f["Grid"]
            current_lon = np.asarray(grid["lon"][:], dtype=np.float64)
            current_lat = np.asarray(grid["lat"][:], dtype=np.float64)

            # Select only the tropical latitudes BEFORE reading the precipitation.
            lat_indices = np.flatnonzero(
                (current_lat >= LAT_MIN) & (current_lat <= LAT_MAX)
            )
            if lat_indices.size == 0 or np.any(np.diff(lat_indices) != 1):
                raise RuntimeError("Unexpected IMERG latitude coordinates.")
            first, stop = lat_indices[0], lat_indices[-1] + 1

            dataset = grid["precipitation"]
            if dataset.shape != (1, current_lon.size, current_lat.size):
                raise RuntimeError(f"Unexpected IMERG dimensions: {dataset.shape}")

            # NASA: (time, lon, lat). Matplotlib: (lat, lon).
            rate = np.asarray(dataset[0, :, first:stop], dtype=np.float32).T
            valid = np.isfinite(rate) & (rate >= 0)

            if "_FillValue" in dataset.attrs:
                valid &= rate != float(np.asarray(dataset.attrs["_FillValue"]).flat[0])

            scale = float(np.asarray(dataset.attrs.get("scale_factor", 1.0)))
            offset = float(np.asarray(dataset.attrs.get("add_offset", 0.0)))
            rate = rate * scale + offset   # mm / hour
            valid &= np.isfinite(rate) & (rate >= 0)

            if accumulated is None:
                lon = current_lon.copy()
                lat = current_lat[first:stop].copy()
                accumulated = np.zeros(rate.shape, dtype=np.float64)
                valid_all = np.ones(rate.shape, dtype=bool)
            elif not (np.array_equal(current_lon, lon) and
                      np.array_equal(current_lat[first:stop], lat)):
                raise RuntimeError("The coordinate grid changed between IMERG files.")

            # Rate [mm/h] x half an hour = accumulation [mm].
            accumulated += np.where(valid, rate, 0.0) * 0.5
            valid_all &= valid

        print(f"  {i:02d}/{N_FILES}: {files[time].name}")

    # A pixel is NaN if any of its 48 half-hours is missing.
    accumulated[~valid_all] = np.nan
    valid_fraction = np.mean(np.isfinite(accumulated))
    print(f"Valid complete pixels: {valid_fraction:.2%}")
    if valid_fraction < 0.85:
        raise RuntimeError("Too many missing pixels; keeping the previous map.")

    # Ensure latitude and longitude are ordered south-to-north and west-to-east.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        accumulated = accumulated[::-1, :]

    lon = ((lon + 180) % 360) - 180
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    accumulated = accumulated[:, lon_order]

    # -------------------------------------------------------------------------
    # 5. CORRECT PIXEL GEOMETRY: centers -> edges
    # -------------------------------------------------------------------------
    # IMERG coordinates refer to pixel CENTERS. pcolormesh(shading="flat")
    # requires pixel EDGES. For a 0.1-degree grid, the edges are +/-0.05 deg.
    # Deriving them from the coordinates avoids a half-pixel map shift.

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

    if accumulated.shape != (lat_edges.size - 1, lon_edges.size - 1):
        raise RuntimeError("The pixel edges do not match the precipitation grid.")

    # -------------------------------------------------------------------------
    # 6. RADAR-STYLE COLOR SCALE
    # -------------------------------------------------------------------------

    levels = [0.1, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    colors = [
        "#e9f7ff", "#a5ddff", "#56b4e9", "#2171b5",
        "#27ae60", "#84cc16", "#ffe34d", "#ffab40",
        "#ff6536", "#ed2929", "#bc1879", "#7c2baf",
    ]
    cmap = ListedColormap(colors)
    cmap.set_under("#ffffff")   # No / negligible rain
    cmap.set_over("#4b136d")    # Very intense accumulation
    cmap.set_bad("#d9dce1")     # Missing data
    norm = BoundaryNorm(levels, cmap.N)

    # -------------------------------------------------------------------------
    # 7. ADMINISTRATIVE BOUNDARIES (NATURAL EARTH)
    # -------------------------------------------------------------------------

    # Downloaded by Cartopy on the first run and cached for later runs.
    # Use the same admin-1 dataset for Mexican states and Colombian departments.
    admin1_path = shapereader.natural_earth(
        resolution="10m", category="cultural",
        name="admin_1_states_provinces",
    )
    admin1_records = list(shapereader.Reader(admin1_path).records())

    # Natural Earth admin-1 records identify their parent country via 'admin'.
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

    # -------------------------------------------------------------------------
    # 8. FIGURES — GLOBAL, AMERICAS, MEXICO, AND COLOMBIA
    # -------------------------------------------------------------------------

    # Prepare all outputs in the same filesystem before publishing them.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".twco_publish_", dir=OUTPUT) as publish_tmp:
        publish_tmp = Path(publish_tmp)

        for title, extent, filename, figsize, lon_step, lat_step in MAPS:
            print(f"Rendering: {title}")

            fig = plt.figure(figsize=figsize, facecolor="white")
            ax = fig.add_axes([0.09, 0.28, 0.82, 0.52],
                              projection=ccrs.PlateCarree())

            mesh = ax.pcolormesh(
                lon_edges, lat_edges, np.ma.masked_invalid(accumulated),
                transform=ccrs.PlateCarree(), shading="flat",
                cmap=cmap, norm=norm, rasterized=True,
            )
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            if filename == "tropical_precip_24h_latest.webp":
                ax.coastlines(resolution="110m", color="#273847", linewidth=0.65)
            else:
                ax.coastlines(resolution="10m", color="#273847", linewidth=0.8)
                ax.add_feature(
                    cfeature.BORDERS.with_scale("10m"),
                    edgecolor="#273847", linewidth=0.85, zorder=5,
                )

                if filename == "mexico_precip_24h_latest.webp":
                    ax.add_geometries(
                        mexico_admin1, crs=ccrs.PlateCarree(),
                        facecolor="none", edgecolor="#404c59",
                        linewidth=0.45, alpha=0.9, zorder=6,
                    )
                elif filename == "colombia_precip_24h_latest.webp":
                    ax.add_geometries(
                        colombia_admin1, crs=ccrs.PlateCarree(),
                        facecolor="none", edgecolor="#404c59",
                        linewidth=0.55, alpha=0.9, zorder=6,
                    )
            ax.set_xticks(np.arange(
                np.ceil(extent[0] / lon_step) * lon_step,
                extent[1] + 0.001, lon_step), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(
                np.ceil(extent[2] / lat_step) * lat_step,
                extent[3] + 0.001, lat_step), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.gridlines(linewidth=0.4, alpha=0.3,
                         color="#555555", linestyle=":")
            ax.tick_params(labelsize=9)

            fig.text(0.055, 0.95, "TROPICAL WEATHER & CLIMATE OBSERVATORY",
                     fontsize=11, weight="bold", color="#1c405b")
            fig.text(0.055, 0.885, title + " | 24-hour Accumulation"
                     if "24-hour" not in title else title,
                     fontsize=19, weight="bold", color="#152a3c")
            fig.text(0.055, 0.83,
                     f"{start:%d %b %Y %H:%M} – {end:%d %b %Y %H:%M} UTC  |  "
                     f"NASA GPM IMERG Early Run {file_version}",
                     fontsize=10, color="#475569")

            # Keep the global tropical colorbar EXACTLY as before.
            if filename == "tropical_precip_24h_latest.webp":
                cax = fig.add_axes([0.15, 0.185, 0.70, 0.032])
                cb = fig.colorbar(
                    mesh, cax=cax, orientation="horizontal",
                    extend="max", ticks=levels,
                )
                cb.set_label("Accumulated precipitation (mm / 24 h)", fontsize=11)
                cb.ax.tick_params(labelsize=9)
            else:
                # Cartopy adjusts the GeoAxes position to preserve map aspect.
                # Force that adjustment before measuring its *actual* width.
                fig.canvas.draw()
                map_box = ax.get_position()

                # Colorbar is always narrower than the actual map (90%).
                bar_width = 0.90 * map_box.width
                bar_left = map_box.x0 + (map_box.width - bar_width) / 2
                bar_height = 0.016  # Half the previous 0.032 figure fraction
                cax = fig.add_axes([bar_left, 0.185, bar_width, bar_height])

                # Keep ALL original color intervals; thin only the labels.
                regional_ticks = [0.1, 2, 10, 30, 75, 150, 300]
                cb = fig.colorbar(
                    mesh, cax=cax, orientation="horizontal",
                    extend="max", ticks=regional_ticks,
                )
                cb.set_label("Accumulated precipitation (mm / 24 h)", fontsize=10)
                cb.ax.tick_params(labelsize=8, length=2, pad=3)

            fig.text(0.055, 0.065,
                     "Source: NASA GPM IMERG Early Run  |  "
                     "DOI: 10.5067/GPM/IMERG/3B-HH-E/07  |  Satellite estimates",
                     fontsize=9, color="#64748b")
            fig.text(0.945, 0.065, "ajaramillomoreno.github.io",
                     fontsize=9, color="#64748b", ha="right")

            # Render at the original 200 dpi and encode losslessly as WebP.
            # The PNG exists only in memory; it is never written to disk.
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=200, facecolor="white")
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.save(publish_tmp / filename, format="WEBP",
                           lossless=True, method=6)
            buffer.close()
            plt.close(fig)

        # ---------------------------------------------------------------------
        # 8. METADATA AND PUBLICATION
        # ---------------------------------------------------------------------
        details = {
            "product": "TWCO-001",
            "title": "Global Tropical Precipitation - 24-hour Accumulation",
            "source": "NASA GPM IMERG Early Run",
            "collection": PRODUCT,
            "collection_version": VERSION,
            "file_version": file_version,
            "doi": "10.5067/GPM/IMERG/3B-HH-E/07",
            "start_utc": start.isoformat(),
            "end_utc_exclusive": end.isoformat(),
            "half_hourly_files": N_FILES,
            "domain": "35S-35N, global",
            "units": "mm / 24 h",
            "valid_pixel_fraction": float(valid_fraction),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "figures": [entry[2] for entry in MAPS],
            "image_format": "webp",
            "compression": "lossless",
        }
        (publish_tmp / JSON.name).write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )

        # Only publish once all four figures and metadata were generated.
        for _, _, filename, _, _, _ in MAPS:
            os.replace(publish_tmp / filename, OUTPUT / filename)
            print(f"Figure saved: {OUTPUT / filename}")
        os.replace(publish_tmp / JSON.name, JSON)
        print(f"Metadata saved: {JSON}")

print("Temporary IMERG data removed. Done.")
