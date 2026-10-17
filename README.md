# MyExperimentToolkit

**Historical optical-image and spectrum analysis utilities.** This repository preserves the public Python toolkit for intensity processing, spectral visualization and sequence handling.

[简体中文](README.zh-CN.md) · [Original documentation and examples](docs/README-history.md)

## Browse the snapshot

| Path | Role in this public version |
|---|---|
| [SensorToolkit/](SensorToolkit/) | Shared loading, processing and visualization components |
| `intensity_analysis.py`, `intensity_analysis-sim.py` | Optical intensity analysis entry scripts |
| `spectral_analysis.py`, `spectral_analysis-sim.py` | Spectral analysis entry scripts |
| `sequence_divider.py`, `sequence_process.py` | Sequence-processing utilities |
| [phase-extrator-app/](phase-extrator-app/) | Separate phase-extraction application material |
| [requirements.txt](requirements.txt) | Original dependency list |

## Status and use

This is the historical **public** snapshot, not a release of a newer private rebuild. No private modules, experimental images, research results or deployment data are included in this documentation update.

The original scripts cover image loading, cropping/filtering, statistics and multidimensional plots. Their assumptions about filenames, wavelength metadata, array layout and local paths must be checked against your own input before use. There is no newly verified one-command installer, packaged release or bundled test dataset. Read the [previous README](docs/README-history.md) and each script first; `requirements.txt` records the original environment dependencies, not a current compatibility guarantee.

No application was launched and no experimental data were read or processed for this documentation pass.

## Rights

**This project does not provide an open-source license.** The original no-open-source-license statement is retained. Public availability does not grant permission to reuse or redistribute the toolkit under MIT or another license. Contact the maintainer for collaboration or reuse terms; dependencies retain their own licenses.
