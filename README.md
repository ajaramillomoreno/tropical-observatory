<div align="center">

<img src="assets/tropical_observatory_logo.png" alt="Tropical Weather & Climate Observatory logo" width="300">

# Tropical Weather & Climate Observatory

**An open observatory of tropical weather, hydroclimate, and climate variability**

*Observing the tropical atmosphere and oceans, with a particular focus on the Americas.*

**Project status:** Under development

</div>

---

## About the observatory

The **Tropical Weather & Climate Observatory (TWCO)** is an open-source project dedicated to monitoring and visualizing the evolving state of the tropical atmosphere and oceans. It aims to make observational data and reproducible scientific diagnostics accessible through regularly updated maps, figures, and time series.

The observatory connects **short-term weather conditions** with **subseasonal-to-interannual climate variability**, emphasizing the atmospheric and oceanic processes that shape tropical hydroclimate. Its primary regional focus is **Mexico, Central America, the Caribbean, and tropical South America**, within a broader tropical and global context when needed to understand large-scale connections.

TWCO is an independent scientific and open-source project developed by **Alejandro Jaramillo Moreno**.

## Scientific scope

The observatory is designed around five complementary themes:

- **Tropical precipitation and hydroclimate:** rainfall accumulations, spatial patterns, anomalies, and wet and dry conditions.
- **Tropical convection:** convective organization and its evolution, including diagnostics based on outgoing longwave radiation (OLR).
- **Atmosphere–ocean interactions:** tropical sea surface temperature anomalies, ENSO, and associated climate variability.
- **Tropical atmospheric dynamics:** circulation patterns, moisture transport, equatorial waves, and large-scale tropical disturbances.
- **Climate variability and extremes:** connections across daily, intraseasonal, seasonal, and interannual timescales, including hydroclimatic extremes.

## Planned observational products

| Product | Main variables or diagnostics | Candidate data source | Update target |
| --- | --- | --- | --- |
| Tropical precipitation | Daily and multi-day rainfall accumulations | NASA GPM IMERG | Daily |
| Tropical convection | OLR fields and anomalies | NOAA OLR datasets | Daily, subject to data availability |
| Tropical oceans | Sea surface temperature and anomalies | NOAA OISST | Daily |
| Atmospheric circulation | Winds and moisture-related diagnostics | Reanalysis products | To be determined |
| Tropical variability | ENSO indicators, Hovmöller diagrams, and equatorial-wave diagnostics | Observations and reanalysis | To be determined |

**These are planned products, not an indication that the automated data pipeline is already operational.** Sources, temporal coverage, refresh schedules, and processing methods will be documented for each product as it is implemented.

## Data sources and scientific transparency

TWCO will rely on openly accessible scientific datasets where possible, including:

- [NASA Global Precipitation Measurement (GPM) / IMERG](https://gpm.nasa.gov/data/imerg)
- [NOAA Optimum Interpolation Sea Surface Temperature (OISST)](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
- [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/) for selected climate datasets and diagnostics

Every published observational product should identify its **original source, valid date or period, units, processing method, and—where applicable—climatological reference period**. Data latency and missing updates will be reported rather than represented as current observations.

Source datasets remain subject to the usage, citation, and redistribution conditions of their respective providers.

## Automation and publication

The intended workflow is:

1. Retrieve the latest available observational data.
2. Validate timestamps, spatial coverage, and data completeness.
3. Compute reproducible diagnostics and generate publication-quality figures in Python.
4. Automatically update published products using **GitHub Actions**.
5. Display the latest products through **GitHub Pages** as the website is developed.

The automation and public dashboard will be introduced incrementally. Until then, this repository serves as the project home and development space.

## Repository organization

The planned structure is:

```text
tropical-observatory/
├── .github/
│   └── workflows/       # Scheduled workflows (planned)
├── assets/
│   └── tropical_observatory_logo.png
├── config/              # Regions and product settings (planned)
├── docs/                # Observatory website (planned)
├── figures/             # Generated figures (planned)
├── src/                 # Data access, diagnostics, plotting (planned)
├── tests/               # Validation and automated tests (planned)
├── .gitignore
├── LICENSE
└── README.md
```

Folders marked *planned* will be added as the corresponding functionality is developed.

## Reproducibility and use

The project will prioritize transparent methods, documented data provenance, open-source scientific computing, and reproducible visualizations. Its figures are intended for **scientific exploration, education, and public communication**; they are **not operational weather forecasts or official hazard warnings**.

For research or publication use, please acknowledge the original data providers and cite the specific TWCO software release or scientific product when citation information becomes available.

## Project lead

**Alejandro Jaramillo Moreno**  
Atmospheric scientist · Tropical climate dynamics · Hydroclimatology

- [Personal website](https://ajaramillomoreno.github.io/)
- [GitHub](https://github.com/ajaramillomoreno)
- [ORCID](https://orcid.org/0000-0002-4175-7726)

## License

The source code is intended to be distributed under the [MIT License](LICENSE). Scientific datasets and third-party materials retain their respective licenses and attribution requirements.
