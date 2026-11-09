# Hybrid Transformer-XGBoost Equity Forecasting Pipeline

This repository contains the codebase for a Master's dissertation project investigating the integration of Time-Series Foundation Models (MOMENT) with gradient boosting frameworks (XGBoost) for financial market forecasting.

The project evaluates whether deep learning embeddings extracted from the MOMENT Foundation Model can improve the directional accuracy and profitability of a Long-Short equity trading strategy compared to traditional statistical (ARIMA) and purely tabular machine learning models.

## Key Features

* **Foundation Model Feature Extraction**: Utilises `momentfm` (MOMENT) to extract high-dimensional representations of financial time series via rolling windows.

* **Hybrid Machine Learning Pipeline**: Combines traditional macroeconomic/technical tabular indicators with transformer-based deep learning embeddings.

* **Robust Hyperparameter Tuning**: Implements Walk-Forward Cross-Validation with purged sequence gaps using `Optuna` to optimize XGBoost on GPU.

* **Explainable AI (XAI)**: Integrates `SHAP` (SHapley Additive exPlanations) to quantify the exact contribution of deep learning embeddings versus traditional indicators.

* **Automated Backtesting & Reporting**: Automatically generates an equity curve comparing the strategy to a Buy & Hold baseline, outputting all plots to a generated HTML report.

## Project Architecture

1. **Data Ingestion (`dataset_generation.py`)**: Fetches historical ticker data (via `yfinance`) and engineers technical/macroeconomic indicators.

2. **Transformer Extraction (`transformer_architecture.py`)**: Uses the MOMENT foundation model to process scaled closing prices in 512-day rolling windows.

3. **Baseline Modeling (`arima_architecture.py`)**: Generates a baseline ARIMA forecast.

4. **XGBoost & Optuna Tuning (`hyperparameter_tuning.py`, `xgb_architecture.py`)**: Trains both a standard Tabular XGBoost model and a Hybrid XGBoost model.

5. **Evaluation & Visualization (`visualisation.py`, `report.py`)**: Computes Hit Rates, ROC AUC, Sharpe ratios, and compiles the metrics into a local web report.

## Prerequisites & Hardware Requirements

This project is highly optimised for **NVIDIA GPUs (CUDA)** due to the memory requirements of the MOMENT transformer and Optuna XGBoost trials.

**Critical Windows Setup:**
If you are running this on a Windows machine, you **must** ensure the following are installed to prevent PyTorch `c10.dll` initialization crashes (WinError 1114):

1. **NVIDIA Graphics Drivers**: Updated to support CUDA 12.4 or higher.

2. **Microsoft Visual C++ Redistributable (X64)**: Required for PyTorch and XGBoost OpenMP C++ backends.

## Dependencies Installation

To install the CUDA version of PyTorch and relevant packages required for the pipeline:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

```

## Usage

The entire pipeline is orchestrated through `main.py` using command-line arguments. By default, running the script with no arguments will backtest the S&P 500 (`^GSPC`) from 2000 to 2026 using the `small` MOMENT model and 50 Hyperparameter trials.

```
python main.py

```

### Custom Execution

You can customise the asset, timeframe, and model constraints using the following arguments:

```
python main.py --ticker "AAPL" --start "2015-01-01" --end "2025-01-01" --n-trials 100 --model_size "AutonLab/MOMENT-1-base"

```

| Argument | Type | Default | Description | 
| ----- | ----- | ----- | ----- | 
| `--ticker` | `str` | `^GSPC` | Yahoo Finance ticker symbol. | 
| `--start` | `str` | `2000-01-01` | Data start date (YYYY-MM-DD). | 
| `--end` | `str` | `2026-01-01` | Data end date (YYYY-MM-DD). | 
| `--n-trials` | `int` | `50` | Number of Optuna hyperparameter tuning trials. | 
| `--model_size` | `str` | `AutonLab/MOMENT-1-small` | Options: `...-small`, `...-base`, `...-large`. | 

*Note: `import torch` must remain on line 1 of `main.py` to manage C++ memory allocation and prevent OS-level DLL lockouts between XGBoost and PyTorch.*

## Outputs & Artifacts

All generated artifacts are saved directly to the `/results` directory.

* **EDA Plots**: Time series decomposition, correlation heatmaps, return distributions.

* **Evaluation Plots**: Confusion matrices, predictions vs. actuals, learning curves.

* **Strategy Analysis**: SHAP summary plots and total equity backtest curves.

* **Final Report**: `visualised_report.html` (A compiled visual summary of the pipeline).