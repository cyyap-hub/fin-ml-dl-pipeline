import torch
import os
import argparse
from src.arima_architecture import arima_model
from src.xgb_architecture import xgboost_model, hybrid_xgboost_model
from src.visualisation import close_over_time, ts_decomposition, log_return_distribution, plot_correlation_heatmap, calculate_and_plot_shap, model_learning_curve, predictions_vs_actuals_plot, confusion_matrix_plot, equity_plot
from src.report import generate_html_report
from src.transformer_architecture import transformer_extraction
from src.backtesting import equity_backtest


def ensure_results():
    os.makedirs("results", exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="^GSPC")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default="2026-01-01")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Hyperparameter trials.")
    parser.add_argument("--model_size", type=str, default="AutonLab/MOMENT-1-small", choices=[
            "AutonLab/MOMENT-1-small",
            "AutonLab/MOMENT-1-base",
            "AutonLab/MOMENT-1-large"
        ], help="HuggingFace model path for MOMENT")
    args = parser.parse_args()

    ensure_results()

    print(f"Plotting Close Prices over time for {args.ticker}...")
    close_over_time()

    print(f"\nBuilding time series decomposition for Close Prices of {args.ticker}...")
    ts_decomposition(ticker=args.ticker, start_date=args.start, end_date=args.end)

    print(f"\nConstructing theoretical distribution and log returns distribution for {args.ticker}...")
    log_return_distribution(ticker=args.ticker, start_date=args.start, end_date=args.end)

    print(f"\nPlotting correlation heatmap for {args.ticker}...")
    plot_correlation_heatmap(ticker=args.ticker, start_date=args.start, end_date=args.end)

    print("\nBuilding Models' Dataframe...")
    models_df=transformer_extraction(ticker=args.ticker, start_date=args.start, end_date=args.end, model_size=args.model_size)

    print(f"\nRunning ARIMA pipeline and plotting model results for {args.ticker}...")
    arima_model(models_df)

    print(f"\nRunning XGBoost pipeline for {args.ticker} using {args.n_trials} trials...")
    xgboost_model(models_df, n_trials=args.n_trials)

    print(f"\nRunning Hybrid pipeline for {args.ticker} using {args.model_size} with {args.n_trials} trials...")
    hybrid_pred_continuous=hybrid_xgboost_model(models_df, n_trials=args.n_trials)

    print(f"\nConstructing Equity Backtest for {args.ticker}...")
    equity_backtest(hybrid_pred_continuous, models_df)

    generate_html_report()

if __name__ == "__main__":
    main()