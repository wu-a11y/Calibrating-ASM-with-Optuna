<<<<<<< HEAD
# DataProcessor

[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

## Project Overview

This project proposes an automated parameter calibration method for the Activated Sludge Model (ASM2d) based on the Optuna framework, aiming to address the inefficiency and expert-dependence issues of traditional manual parameter tuning. By introducing the Tree-structured Parzen Estimator (TPE) and Non-dominated Sorting Genetic Algorithm II (NSGA-II) from Optuna, the model automatically optimizes key wastewater treatment parameters such as Total Nitrogen (TN) and Chemical Oxygen Demand (COD). The results show that TPE significantly improves the prediction accuracy of TN and COD in single-objective optimization, reducing tuning errors and iterations. On the other hand, NSGA-II optimizes both TN and COD simultaneously in multi-objective optimization, significantly reducing the errors of both parameters. Additionally, the Optuna-based approach enhances calibration efficiency, reducing tuning computational time and iteration cycles. This framework provides a novel solution for automated and intelligent wastewater treatment modeling and optimization, with good practical applicability and potential for wider adoption.

## Project Structure

```
QSDsan
├── code
│   └── main.py
├── data
│   ├── process
│   ├── raw
│   │   ├── input.xlsx
│   │   ├── COD_OSA_FP.xlsx
│   │   ├── MO_OSA_FP.xlsx
│   │   ├── TN_OSA_FP.xlsx
│   │   ├── Water_Treatment_Data.xlsx
│   │   └── Traditional sensitivity analysis
│   │       ├── 1 Traditional sensitivity analysis.xlsx
│   │       ├── 2 Traditional sensitivity analysis.xlsx
│   │       ├── ......
│   │       └── 51 Traditional sensitivity analysis.xlsx
│   └── result
├── docs
│   ├── Akiba et al. - 2019 - Optuna A Next-generation Hyperparameter Optimizat.pdf
│   │── Li et al. - 2022 - QSDsan an integrated platform for quantitative su.pdf
│   └── site-packages
│       ├── __pycache__
│       ├── _argon2_cffi_bindings
│       ├── _distutils_hack
│       ├── ......
│       └── zope.interface-5.4.0-py3.11-nspkg.pth
├── output
├── picture
├── README.md
└── requirements.txt


```

---
## Development Guide

### Project Architecture

This project is a wastewater treatment system simulation and parameter optimization project based on `qsdsan` and `optuna`. It is mainly used for simulating the wastewater treatment process and optimizing related parameters to reduce the discrepancies between actual effluent and target effluent values in COD, TN, TP, and other indicators. The project structure can be analyzed from two perspectives: functional modules and process logic.

1. **Influent Concentration Adjustment Module**: Achieved through the `Adjusting_influent_concentration` function, which dynamically adjusts the initial values of corresponding indicators (e.g., `X_S` for COD, `S_NO3` for TN) in the influent composition based on the target COD, TN, and TP concentrations to ensure that the simulated influent concentration matches the target value.

2. **Wastewater Treatment System Simulation Module**: The core function is `simulate_one`, which is responsible for constructing the wastewater treatment system and running the simulation:

   * Defines parameters for influent, effluent, return flows, and other logistics (e.g., `WasteStream`), including flow rate, temperature, etc.
   * Builds the reactor model: including anoxic zones (A1, A2), aerobic zones (O1, O2, O3), and the clarifier (C1), and configures aeration models (`DiffusedAeration`) and biological kinetic models (`ASM2d`).
   * Initializes system states (loads initial concentration data from Excel), sets simulation time, time steps, and integration methods, and executes dynamic simulations.
   * Calculates the percentage differences between simulated results and target values (COD, TN, TP) and stores the results in an Excel file.

3. **Parameter Optimization Module**: Includes both single-objective and multi-objective optimization, based on the intelligent optimization (OT) in `optuna` and traditional trial-and-error methods (TT):

   * **Single-objective optimization**: `OT` (Optuna optimization) and `TT` (trial-and-error method) functions, each minimizing the percentage difference of COD or TN, searching for optimal parameters.
   * **Multi-objective optimization**: `MO_OT` (multi-objective Optuna optimization) and `MO_TT` (multi-objective trial-and-error) functions, simultaneously minimizing the percentage difference of COD, TN, and TP, aiming to find the Pareto-optimal solution.

4. **Sensitivity Analysis Module**: Achieved through `OSA` and `TSA` functions, which read sensitivity analysis data from Excel files and filter out the parameters with the greatest impact on target indicators (such as COD and TN), which are then used as candidates for optimization.

5. **Data Storage and Visualization Module**:

   * Stores simulation results, optimized parameters, and sensitivity analysis results in Excel files, categorized by date, day, and experiment number (e.g., path `../data/process/{name}/{day}/`).
   * Provides visualization capabilities via `optuna` (e.g., optimization history, parameter correlations), which can be exported as HTML files (currently commented out in the code).

### Process Logic

1. **Parameter Preparation**: Selects key parameters from the sensitivity analysis data using `OSA` or `TSA` functions, determining initial values.
2. **Influent Adjustment**: Calls the `Adjusting_influent_concentration` function to ensure the simulated influent concentration matches the target values.
3. **System Simulation**: Uses the `simulate_one` function to build the wastewater treatment system, run simulations, and compute differences from the target values.
4. **Parameter Optimization**: Based on the need, either single-objective (`OT`/`TT`) or multi-objective (`MO_OT`/`MO_TT`) optimization methods are chosen to search for optimal parameters based on simulation results.
5. **Result Saving**: Writes optimized parameters and target differences into Excel files, completing one cycle of simulation-optimization.

The overall architecture is designed around the "simulation-optimization-analysis" closed loop, with modular functions decoupling various stages, allowing flexible adjustment of input parameters, optimization goals, and storage paths. This structure is suitable for wastewater treatment system parameter tuning and performance optimization in different scenarios.

---

## Features

* **Feature 1**: The project implements automated parameter optimization for the Activated Sludge Model (ASM2d) using the Optuna framework, supporting single-objective and multi-objective optimization, which improves model accuracy and tuning efficiency.

* **Feature 2**: The innovation of this project lies in integrating machine learning hyperparameter optimization techniques with wastewater treatment models, supporting full parameter optimization and multi-objective optimization, significantly improving optimization efficiency and accuracy.

* **Feature 3**: The project is widely applicable in wastewater treatment plants, optimizing pollutant emissions and treatment processes. It offers practical value by improving treatment efficiency, reducing costs, and promoting environmental protection.

---

## Installation and Configuration

### System Requirements

* Python 3.11

### Installation Steps

```bash
# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

* `qsdsan`: Used to build wastewater treatment units, logistics, and biological kinetic models (e.g., ASM2d).
* `optuna`: Used for parameter optimization, supporting single-objective/multi-objective search and sensitivity analysis.
* `pandas`/`numpy`: For data processing and numerical computation.
* `matplotlib`/`plotly`: For result visualization (e.g., charts, optimization process visualization).

---

## Usage

### Basic Usage

```bash
cd code
python main.py
```

### sample code

```python
import qsdsan as qs
import optuna
import numpy as np
from qsdsan import sanunits as su, processes as pc, WasteStream, System
from qsdsan.utils import time_printer, load_data, get_SRT
import warnings
import pandas as pd
import plotly
import plotly.io as pio
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import matplotlib
matplotlib.use('Agg')  
from datetime import datetime
import os
import time
import optuna
warnings.filterwarnings('ignore') 
import psycopg2


def Adjusting_influent_concentration(first_inflow_value, input_COD_value, input_TN_value, input_TP_value)

def simulate_one(first_inflow_value, external_return_flow_value,
             input_COD_value, output_COD_value, adjusting_COD_value,
             input_TN_value, output_TN_value, adjusting_TN_value,
             input_TP_value, output_TP_value, adjusting_TP_value,
             param_names, param_initial_values, parameter_adjustments, day, name, trial
             )
   
def OT(param_names, param_initial_values, day, name_1)
def TT(param_names, param_initial_values, day, name_1)
def MO_OT(param_names, param_initial_values, day, name_1)
def MO_TT(param_names, param_initial_values, day, name_1)
   
def OSA(name_1)
def TSA(name_1)
def FP(name_1)

   name_1 = 'MO'
   name_2 = 'FP'
   name_3 = 'MO_OT'
   
   param_names, param_initial_values = param_function(name_1)
   tuning_function(param_names, param_initial_values, day, name_1)
```

### function declaration
| Function                             | Description                                |
| ------------------------------------ | ------------------------------------------ |
| `Adjusting_influent_concentration()` | Correction function                        |
| `simulate_one()`                     | Simulation function                        |
| `OT()`                               | Optuna single-objective tuning method      |
| `TT()`                               | Traditional single-objective tuning method |
| `MO_OT()`                            | Optuna multi-objective tuning method       |
| `MO_TT()`                            | Traditional multi-objective tuning method  |
| `OSA()`                              | Optuna parameter selection                 |
| `TSA()`                              | Traditional parameter selection            |
| `FP()`                               | Full parameter selection                   |


---

## Frequently Asked Questions

1. **Question 1**: How to resolve dependency errors at startup?
   **Answer**: Ensure all dependencies are properly installed. You can try recreating the virtual environment.  
2. **Question 2**: How to solve the problem of not being able to find the downloaded installation package version?  
   **Answer**: Please check the installation packages in the "docs/site-packages" directory to ensure that the version you have installed is correct.

---

## Contact Information

* Project Maintainer: Yue Wang
* Contact Email: [642544234@qq.com](mailto:642544234@qq.com)
* Project URL: [https://github.com/janedoe/dataprocessor](https://github.com/janedoe/dataprocessor)

---
Last Updated: 2025-07-28

=======
