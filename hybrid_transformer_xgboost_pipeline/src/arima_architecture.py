import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from dataset_generation import build_dataset
from visualisation import predictions_vs_actuals_arima

def arima_model(df):
    """ARIMA training and testing."""
    # Load Data
    y = df['Target_Log_Return']
    print(f"Total Target Vector (y) Shape: {y.shape}")

    # Training and Testing Set Split
    print("\nIsolating the Out-of-Sample Test Set (80/20 Split)...")
    split_idx = int(len(y) * 0.8)

    y_optimisation = y.iloc[:split_idx]
    y_test_final = y.iloc[split_idx:]

    print(f"Optimisation Set (80%): {y_optimisation.shape}")
    print(f"True OOS Test Set (20%): {y_test_final.shape}")


    # Out-of-Sample Evaluation
    print("\n[ARIMA] Training Model...")
    final_arima_model = auto_arima(
        y_optimisation.values,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=0, seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True
    )

    arima_params = final_arima_model.get_params()
    best_order = arima_params.get('order')
    print(f"The optimal (p, d, q) order is: {best_order}")

    print("[ARIMA] Forecasting Out-of-Sample Test Set (Final 20%)...")
    final_forecast = final_arima_model.predict(n_periods=len(y_test_final))

    final_results = pd.DataFrame(
    {"Actual": y_test_final.values, "ARIMA_Pred": final_forecast},
        index=y_test_final.index
    )

    y_true = final_results['Actual'].values
    y_pred = final_results['ARIMA_Pred'].values

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate Directional Accuracy
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    correct_guesses = (true_direction == pred_direction) & (true_direction != 0)
    hit_rate = correct_guesses.sum() / len(y_true)

    print(f"\n=== ARIMA BASELINE METRICS ===")
    print(f"RMSE:                 {rmse:.6f}")
    print(f"MAE:                  {mae:.6f}")
    print(f"Directional Accuracy: {hit_rate * 100:.2f}%")
    print("==================================================\n")

    predictions_vs_actuals_arima(final_results, y_true, y_pred, rmse, mae, hit_rate)