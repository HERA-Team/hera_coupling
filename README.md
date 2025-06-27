# hera_coupling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Python package for modeling and mitigating antenna-to-antenna mutual coupling in HERA visibility data.

## Features

- **Coupling Parameter Storage**: Store coupling parameters in a `UVMutualCoupling` object with metadata (`antpos`, `freqs`, `pols`, `times`).
- **Forward & Inverse Coupling**: Perform both forward coupling and inverse decoupling operations, with optional first-order and multi-path corrections.
- **HDF5 Read/Write**: Read/write coupling parameters to HDF5 files via `UVMutualCoupling.write_coupling` and `UVMutualCoupling.write_coupling`.
- **Multiple Data Format Support**: Support for both `hera_cal.datacontainer.DataContainer` and `numpy.ndarray`

## Installation

```
pip install git+https://github.com/HERA-Team/hera_coupling.git
```

or 

```
git clone https://github.com/HERA-Team/hera_coupling.git
cd hera_coupling
pip install -e .
```

## Requirements

- Python >= 3.10
- `numpy`
- `scipy`
- `h5py`
- `hera_calibration`