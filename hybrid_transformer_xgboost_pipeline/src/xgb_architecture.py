import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
from hyperparameter_tuning import objective
from visualisation import calculate_and_plot_shap, model_learning_curve, predictions_vs_actuals_plot, confusion_matrix_plot

def xgboost_model(df, n_trials=50):
    # Selection of Plain XGBoost model's Features and Targets
    vanilla_xgboost_feature_columns = ['Volume', 'RSI', 'MACD', 'BB_Lower', 'BB_Middle', 'BB_Upper', 'VIX_Close',
                                       '10Y_Yield', 'Log_Return_Lag1', 'Log_Return_Lag2', 'Log_Return_Lag3',
                                       'Log_Return_Lag4', 'Log_Return_Lag5', 'Log_Return_Lag30']
    X = df[vanilla_xgboost_feature_columns]
    y = df['Target_Log_Return']

    model_name = "XGBoost"
    print(f"{model_name} Model Feature Matrix (X) Shape: {X.shape}")
    print(f"{model_name} Model Target Vector (y) Shape: {y.shape}")

    # Hyperparameter Selection
    print('\nLaunching Optuna Study on GPU...')
    # Use a seeded TPESampler to ensure the sequence of hyperparameters guessed is exactly the same each run
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', study_name='XGBoost_Tuning', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X, y, ), n_trials=n_trials)

    print("\n=== OPTIMISATION COMPLETE ===")
    print(f"Best Directional Accuracy: {study.best_value * 100:.2f}%")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # XGBOOST MODEL FITTING AND EVALUATION
    print(f"\n--- Training Final {model_name} Model ---")

    # Split into 80% train, 20% for out-of-sample test
    split_index = int(len(X) * 0.8)
    X_train_final, X_test_final = X.iloc[:split_index], X.iloc[split_index:]
    y_train_final, y_test_final = y.iloc[:split_index], y.iloc[split_index:]

    xgb_dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
    xgb_dtest = xgb.DMatrix(X_test_final, label=y_test_final)

    # Setup evaluation list and dictionary to capture loss history
    xgb_evals = [(xgb_dtrain, ' Train'), (xgb_dtest, 'Validation')]
    xgb_evals_result = {}

    # Configure final model using the exact best parameters
    final_params = study.best_params.copy()

    # Train the model
    print(f"\n--- Training {model_name} and Generating Learning Curve ---")

    xgb_model = xgb.train(
        final_params,
        xgb_dtrain,
        num_boost_round=1000,
        evals=xgb_evals,
        evals_result=xgb_evals_result,
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # Extract the RMSE histories
    train_rmse = xgb_evals_result['Train']['rmse']
    test_rmse = xgb_evals_result['Validation']['rmse']
    xgb_epochs = len(train_rmse)
    x_axis = range(0, xgb_epochs)

    model_learning_curve(model_name, xgb_model, train_rmse, test_rmse, x_axis)

    # To Generate Out-of-Sample Predictions using the best iteration
    xgb_dtest_final = xgb.DMatrix(X_test_final)
    xgb_pred_continuous = xgb_model.predict(xgb_dtest_final,
                                                  iteration_range=(0, xgb_model.best_iteration + 1))

    # Convert continuous returns to binary classes (1 for UP, 0 for DOWN)
    y_true_binary = (y_test_final.values > 0).astype(int)
    xgb_pred_binary = (xgb_pred_continuous > 0).astype(int)
    # Calculate Hit Rate & ROC AUC
    hit_rate = np.mean((y_true_binary == xgb_pred_binary) & (y_test_final.values != 0))
    auc_score = roc_auc_score(y_true_binary, xgb_pred_continuous)

    # TEST RESULT PLOTS
    print(f"\n--- Plotting {model_name} Predictions vs Actuals ---")
    predictions_vs_actuals_plot(model_name, y_test_final, xgb_pred_continuous, y_true_binary, xgb_pred_binary,
                                hit_rate, auc_score)

    print(f"\n--- Constructing {model_name} Confusion Matrix ---")

    confusion_matrix_plot(model_name, y_true_binary, xgb_pred_binary, hit_rate, auc_score)

def hybrid_xgboost_model(df, n_trials=50):
    # Dropping non-stationary Close prices, the Target before assigning to X
    cols_to_drop = ['Close', 'Target_Log_Return']

    X = df.drop(columns=cols_to_drop)
    y = df['Target_Log_Return']
    model_name="Hybrid"
    print(f"{model_name} Model Feature Matrix (X) Shape: {X.shape}")
    print(f"{model_name} Model Target Vector (y) Shape: {y.shape}")

    # Hyperparameter Selection
    print('\nLaunching Optuna Study on GPU...')
    # Use a seeded TPESampler to ensure the sequence of hyperparameters guessed is exactly the same each run
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', study_name='Hybrid_XGBoost_Tuning', sampler=sampler)
    study.optimize(lambda trial: objective(trial, X, y,), n_trials=n_trials)

    print("\n=== OPTIMISATION COMPLETE ===")
    print(f"Best Directional Accuracy: {study.best_value * 100:.2f}%")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")


    # HYBRID MODEL FITTING AND EVALUATION
    print(f"\n--- Training Final {model_name} Model ---")

    # Split into 80% train, 20% for out-of-sample test
    split_index = int(len(X) * 0.8)
    X_train_final, X_test_final = X.iloc[:split_index], X.iloc[split_index:]
    y_train_final, y_test_final = y.iloc[:split_index], y.iloc[split_index:]


    hybrid_dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
    hybrid_dtest = xgb.DMatrix(X_test_final, label=y_test_final)

    # Setup evaluation list and dictionary to capture loss history
    hybrid_evals = [(hybrid_dtrain,' Train'),(hybrid_dtest,'Validation')]
    hybrid_evals_result = {}

    # Configure final model using the exact best parameters
    final_params = study.best_params.copy()

    # Train the model
    print(f"\n--- Training {model_name} and Generating Learning Curve ---")

    hybrid_model = xgb.train(
        final_params,
        hybrid_dtrain,
        num_boost_round=1000,
        evals=hybrid_evals,
        evals_result=hybrid_evals_result,
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # Extract the RMSE histories
    train_rmse = hybrid_evals_result['Train']['rmse']
    test_rmse = hybrid_evals_result['Validation']['rmse']
    hybrid_epochs = len(train_rmse)
    x_axis = range(0, hybrid_epochs)

    model_learning_curve(model_name, hybrid_model, train_rmse, test_rmse, x_axis)

    # To Generate Out-of-Sample Predictions using the best iteration
    hybrid_dtest_final = xgb.DMatrix(X_test_final)
    hybrid_pred_continuous = hybrid_model.predict(hybrid_dtest_final,iteration_range=(0, hybrid_model.best_iteration + 1))

    # Convert continuous returns to binary classes (1 for UP, 0 for DOWN)
    y_true_binary = (y_test_final.values > 0).astype(int)
    hybrid_pred_binary = (hybrid_pred_continuous > 0).astype(int)
    # Calculate Hit Rate & ROC AUC
    hit_rate = np.mean((y_true_binary == hybrid_pred_binary) & (y_test_final.values != 0))
    auc_score = roc_auc_score(y_true_binary, hybrid_pred_continuous)

    # TEST RESULT PLOTS
    print(f"\n--- Plotting {model_name} Predictions vs Actuals ---")
    predictions_vs_actuals_plot(model_name,y_test_final, hybrid_pred_continuous, y_true_binary, hybrid_pred_binary, hit_rate, auc_score)

    print(f"\n--- Constructing {model_name} Confusion Matrix ---")

    confusion_matrix_plot(model_name, y_true_binary, hybrid_pred_binary, hit_rate, auc_score)

    print(f"\n--- Creating SHAP Analysis Plots ---")
    calculate_and_plot_shap(hybrid_model, X_test_final)

    return hybrid_pred_continuous



