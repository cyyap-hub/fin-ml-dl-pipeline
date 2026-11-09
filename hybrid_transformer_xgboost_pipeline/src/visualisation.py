import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")
from dataset_generation import build_dataset
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

def close_over_time(ticker="^GSPC", start_date="2000-01-01", end_date="2026-08-01"):
    dataset=build_dataset(ticker=ticker,start_date=start_date,end_date=end_date)
    # Drop NAs from dataset
    dataset=dataset.dropna()

    # Extracting Close Prices
    Close = dataset[['Close']]

    # Create the plot canvass
    plt.figure(figsize=(14, 7)) # Wide, cinematic aspect ratio

    # Setting up the font and theme of our plot
    sns.set_theme(style="ticks") # Use a clean, modern grid style
    plt.rcParams['font.family'] = 'sans-serif' # Ensure a clean font

    # Plot the closing price using seaborn's lineplot for smooth rendering
    sns.lineplot(
        data=Close,
        x=Close.index,
        y='Close',
        color='#0b57d0',
        linewidth=2
    )

    # Formatting the plot aesthetics
    plt.title("Ticker Close Price History", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=14, labelpad=10)
    plt.ylabel("Closing Price (USD)", fontsize=14, labelpad=10)

    # Add a subtle background fill under the line for depth
    plt.fill_between(
        Close.index,
        Close['Close'],
        alpha=0.1,
        color='white'
    )

    # Customize the grid and ticks
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # Remove the top and right borders for a cleaner look
    sns.despine()

    # Ensure everything fits perfectly
    plt.tight_layout()

    # Save and show the plot
    filename = "results/ticker_closing_price.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as '{filename}'")
    plt.close()

def ts_decomposition(ticker="^GSPC", start_date="2000-01-01", end_date="2026-08-01"):
    # STL DECOMPOSITION (Seasonal and Trend decomposition using Loess)
    dataset=build_dataset(ticker=ticker,start_date=start_date,end_date=end_date)
    dataset=dataset.dropna()

    Close = dataset[['Close']]
    # We instantiate the STL object, passing in the data and the cycle length (period).
    # 'robust=True' tells the math to ignore wild outliers when smoothing.
    # 'seasonal=13' is the "stiffness" window. (7 or 13 are standard)
    stl_model = STL(Close, period=12, seasonal=13, robust=True)

    # Run the fit algorithm
    stl_result = stl_model.fit()

    # Plotting decompositions
    fig = stl_result.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle('Closing Price Decomposition (Trend, Seasonality, Residuals)', fontsize=16)

    # Adjust layout to make room for the title
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Adjust layout to make room for the title
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save the figure as a high-resolution PNG
    fig.savefig('results/stl_decomposition_plot.png', bbox_inches='tight', dpi=300)
    plt.close()


def log_return_distribution(ticker="^GSPC", start_date="2000-01-01", end_date="2026-08-01"):
    log_return_df=build_dataset(ticker=ticker,start_date=start_date,end_date=end_date)
    log_return_df=log_return_df.dropna()
    # Calculate Statistics with Target_Log_Return
    log_returns = log_return_df['Target_Log_Return'].values
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    kurtosis = log_return_df['Target_Log_Return'].kurtosis()
    skewness = log_return_df['Target_Log_Return'].skew()

    print(f"Mean (mu): {mu:.6f}")
    print(f"Standard Deviation (sigma): {sigma:.6f}")
    print(f"Kurtosis: {kurtosis:.2f} (Normal distribution = 3.0)")

    # Setting up themes and aesthetics for the plot
    sns.set_theme(style="ticks") # Using a clean, modern grid style
    plt.rcParams['font.family'] = 'sans-serif' # Ensure a clean font
    plt.figure(figsize=(12, 6))

    # Plot the Histogram of actual log returns
    sns.histplot(log_returns, bins=100, stat='density', color='steelblue', alpha=0.6, label='Actual Log Returns')


    # Overlay a Theoretical Normal Distribution (Bell Curve) for comparison
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma) # Create the bell curve using our mean and std dev
    plt.plot(x, p, 'k', linewidth=2, linestyle='--', label='Theoretical Normal Dist.')

    # Amend labels for our Log Return Distribution
    plt.title('Distribution of Daily Log Returns (Jan 2000 to July 2026)', fontsize=16, pad=15)
    plt.xlabel('Daily Log Return', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # Add a text box with statistics
    stats_text = (
    f"Observations: {len(log_returns)}\n"
    f"Mean ($\mu$): {mu:.6f}\n"
    f"Std Dev ($\sigma$): {sigma:.6f}\n"
    f"Skewness: {skewness:.2f}\n"
    f"Kurtosis: {kurtosis:.2f}" # Note: Pandas calculates excess kurtosis natively
    )
    plt.annotate(
    stats_text,
    xy=(0.02, 0.95), xycoords='axes fraction',
    fontsize=11, fontfamily='monospace',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    )

    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xlim(-0.06, 0.06) # Zoom in to ignore extreme outliers for better visual clarity
    plt.tight_layout()

    # Save the plot
    plt.savefig('results/log_returns_distribution.png', dpi=300)
    print("Saved plot as 'log_returns_distribution.png'")
    plt.close()

def plot_correlation_heatmap(ticker="^GSPC", start_date="2000-01-01", end_date="2026-08-01"):
    # Select all the features for the heatmap
    dataset=build_dataset(ticker=ticker,start_date=start_date,end_date=end_date)
    df=dataset.dropna()
    features_to_plot = [
        'VIX_Close', '10Y_Yield', 'RSI', 'MACD',
        'BB_Lower', 'BB_Middle', 'BB_Upper', 'Volume',
        'Target_Log_Return', 'Log_Return_Lag1', 'Log_Return_Lag2',
        'Log_Return_Lag3', 'Log_Return_Lag4', 'Log_Return_Lag5', 'Log_Return_Lag30',
    ]
    df_final = df[features_to_plot].dropna()

    # Calculate the Pearson correlation matrix
    corr_matrix = df_final.corr()

    # Create a mask to hide the upper triangle
    # This prevents showing duplicate information across the diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")

    #  Generate a custom diverging colormap (Blue for negative, Red for positive)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,          # Show the actual correlation numbers
        fmt=".2f",           # Format to 2 decimal places
        square=True,         # Make cells perfectly square
        linewidths=.5,       # Add gridlines between cells
        cbar_kws={"shrink": .75} # Shrink the colorbar slightly
    )

    # Specifying format labels and aesthetics
    plt.title('Pearson Correlation Heatmap: Tabular Features & Target Variable',
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig('results/tabular_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved successfully as 'tabular_correlation_heatmap.png'")
    plt.close()

def predictions_vs_actuals_arima(results_df, y_true, y_pred, rmse, mae, hit_rate):
    # Plot the ARIMA Results
    plt.figure(figsize=(14, 7))
    plt.plot(results_df.index, y_true, label='Actual Returns', color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(results_df.index, y_pred, label='ARIMA Forecast', color='red', alpha=0.8, linewidth=2)
    plt.title(f"Out-of-Sample Validation: ARIMA Model", fontsize=16, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Log Returns", fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    metrics_text = (
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        f"Directional Acc: {hit_rate * 100:.2f}%"
    )
    plt.annotate(
        metrics_text,
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9), # Shrunk to match Master Pipeline
        verticalalignment='top'
    )

    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/arima_predictions_vs_actuals.png", dpi=300)

def model_learning_curve(model_name, model, train_rmse, test_rmse, x_axis):
    #  Model Learning Plot
    plt.figure(figsize=(10, 6))

    # Plot Training Error
    plt.plot(x_axis, train_rmse, label='Training RMSE', color='#1f77b4', linewidth=2)

    # Plot Validation Error
    plt.plot(x_axis, test_rmse, label='Validation RMSE', color='#ff7f0e', linewidth=2)

    # Highlight the point where Early Stopping triggered (the best iteration)
    model_best_iteration = model.best_iteration
    model_best_score = test_rmse[model_best_iteration]

    plt.axvline(x=model_best_iteration, color='gray', linestyle='--', alpha=0.7)
    plt.scatter(model_best_iteration, model_best_score, color='red', s=50, zorder=5, label=f'Best Iteration ({model_best_iteration})')

    # Formatting
    plt.title(f"{model_name} Model Learning Curve", fontsize=16, pad=15, fontweight='bold')
    plt.xlabel("Number of Boosting Rounds (Trees)", fontsize=12)
    plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.legend(loc="upper right", fontsize=11)

    # Remove top/right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filename = f"results/{model_name}_model_learning_curve.png"
    plt.savefig(filename, dpi=300)
    print(f"-> Saved Learning Curve to '{filename}'")
    plt.close()

def predictions_vs_actuals_plot(model_name, actuals, predictions, true_binary, pred_binary, hit_rate, auc_score):
    # Calculate RMSE & MAE
    rmse = np.sqrt(mean_squared_error(actuals.values, predictions))
    mae = mean_absolute_error(actuals.values, predictions)

    print("\n=== HYBRID MODEL OUT-OF-SAMPLE METRICS ===")
    print(f"RMSE:                 {rmse:.6f}")
    print(f"MAE:                  {mae:.6f}")
    print(f"Directional Accuracy: {hit_rate * 100:.2f}%")
    print(f"ROC AUC Score:        {auc_score:.4f}")
    print("==========================================\n")

    # Continuous Line Chart
    plt.figure(figsize=(14, 7))
    plt.plot(actuals.index, actuals.values, label='Actual Returns', color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(actuals.index, predictions, label=f'{model_name} Forecast', color='blue', alpha=0.8, linewidth=1.5)

    plt.title(f"Out-of-Sample Validation: {model_name}", fontsize=16, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Log Returns", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add continuous and directional metrics box
    metrics_text = (
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        )
    plt.annotate(
        metrics_text,
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9),
        verticalalignment='top'
        )

    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_predictions_plot.png", dpi=300)
    print(f"-> Saved '{model_name}_predictions_plot.png'")
    plt.close()

def confusion_matrix_plot(model_name, y_true_binary, pred_binary, hit_rate, auc_score):
    # Confusion Matrix
    cm = confusion_matrix(y_true_binary, pred_binary)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
    disp.plot(cmap='Blues', ax=ax)

    plt.title(f"Out-of-Sample Confusion Matrix: {model_name} Model", fontsize=14, pad=15)

    cm_metrics_text = (f"Directional Accuracy: {hit_rate * 100:.2f}%\n"
                   f"ROC AUC Score: {auc_score:.4f}\n"
    )

    plt.annotate(cm_metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, fontfamily='monospace', bbox=dict(boxstyle="round, pad=0.5", fc="w", edgecolor= "black", alpha=0.9), verticalalignment='top'
    )

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_confusion_matrix.png", dpi=300)
    print(f"-> Saved '{model_name}_confusion_matrix.png'")
    plt.close()

# SHAP values computation and plot
def calculate_and_plot_shap(best_model, X_test, max_display=15):
    import shap
    print("\n[SHAP] Calculating SHAP values on out-of-sample test set...")
    # Initialise TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test)

    # Standard Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Top Individual Features)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("results/shap_summary_plot.png", dpi=300)
    print("-> Saved 'shap_summary_plot.png'")
    plt.close()

    # Aggregate MOMENT Embeddings vs Tabular Features
    abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_test.columns

    is_embedding = [c.startswith("emb_") for c in feature_names]

    moment_importance = abs_shap[is_embedding].sum()
    tabular_importance = abs_shap[~np.array(is_embedding)].sum()

    print("\n--- SHAP Importance Breakdown ---")
    print(f"Total Combined Extracted Features SHAP Weight: {moment_importance:.6f}")
    print(f"Total Combined Technical and Macroeconomic Features SHAP Weight: {tabular_importance:.6f}")

    # Plot the comparison Bar Chart
    labels = ['Extracted Features', 'Technical and Macroeconomic Indicators']
    values = [moment_importance, tabular_importance]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e'], edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(values) * 0.01),
                 f"{yval:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title("Total SHAP Contribution:\nExtracted Features vs. Macroeconomic Features and Technical Indicators", fontsize=14, fontweight='bold',
              pad=15)
    plt.ylabel("Sum of Mean Absolute SHAP Values", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/total_shap.png", dpi=300)
    print("-> Saved 'total_shap.png'")
    plt.close()

def equity_plot(y_test_final, cum_market_returns, cum_hybrid_strategy_returns, market_sharpe, hybrid_strategy_sharpe):
    # Plotting the Equity Curves
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_final.index, cum_market_returns * 100, label='Buy & Hold (S&P 500)', color='gray', alpha=0.7, linewidth=1.5)
    plt.plot(y_test_final.index, cum_hybrid_strategy_returns * 100, label='Hybrid Long-Short Strategy', color='#146c2e', alpha=0.9, linewidth=2) # Dark green

    plt.title("Cumulative Long-Short Strategy Profit & Loss vs. Buy & Hold Baseline", fontsize=16, pad=15, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.5)

    # Add Metrics Text Box to the chart
    metrics_text = (
    f"Hybrid Total Return:{cum_hybrid_strategy_returns[-1]*100:>6.2f}%\n"
    f"Market Total Return:{cum_market_returns[-1]*100:>6.2f}%\n"

    f"Hybrid Strategy Sharpe:{hybrid_strategy_sharpe:>6.2f}\n"
    f"Market Sharpe:{market_sharpe:>6.2f}"
    )
    plt.annotate(
    metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
    fontsize=12, fontfamily='monospace',
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9),
    verticalalignment='top'
    )

    # Position the legend so it doesn't overlap the metrics box
    plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.75), fontsize=12)

    # Remove top and right spines for a cleaner academic look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/equity_curve_plot.png", dpi=300)
    print("-> Saved 'equity_curve_plot.png'")
    plt.close()

if __name__=="__main__":
    close_over_time()
    ts_decomposition()
    log_return_distribution()
    plot_correlation_heatmap()