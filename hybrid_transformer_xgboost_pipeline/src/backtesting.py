import numpy as np
import matplotlib.pyplot as plt
from xgb_architecture import hybrid_xgboost_model
from dataset_generation import build_dataset
from visualisation import equity_plot


def equity_backtest(hybrid_pred_continuous, df):
    print("\n--- Generating Equity Curve (Strategy vs Buy & Hold) ---")
    y=df['Target_Log_Return']
    split_index = int(len(y) * 0.8)
    y_test_final=y.iloc[split_index:]

    # Generate Trading Signals
    # If prediction > 0, go Long (+1). If prediction < 0, go Short (-1).
    hybrid_signals = np.sign(hybrid_pred_continuous)

    # Calculate Daily Returns
    # The return of the strategy is our signal multiplied by what the market actually did
    hybrid_strategy_returns = hybrid_signals * y_test_final.values

    # Calculate Cumulative Geometric Returns
    # We use exp(cumsum(log_returns)) - 1 to obtain the true compounding percentage return
    cum_market_returns = np.exp(np.cumsum(y_test_final.values)) - 1
    cum_hybrid_strategy_returns = np.exp(np.cumsum(hybrid_strategy_returns)) - 1

    # Calculate Financial Metrics (Annualized Sharpe Ratio)
    # Assumes 252 trading days in a year
    market_sharpe = (np.mean(y_test_final.values) / np.std(y_test_final.values)) * np.sqrt(252) if np.std(y_test_final.values) > 0 else 0
    hybrid_strategy_sharpe = (np.mean(hybrid_strategy_returns) / np.std(hybrid_strategy_returns)) * np.sqrt(252) if np.std(hybrid_strategy_returns) > 0 else 0

    equity_plot(y_test_final, cum_market_returns, cum_hybrid_strategy_returns, market_sharpe, hybrid_strategy_sharpe)
