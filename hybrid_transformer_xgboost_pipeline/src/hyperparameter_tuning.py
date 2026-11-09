import optuna
import xgboost as xgb

def objective(trial, X, y, seq_len=512):
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np

    def walk_forward_cv(params, X, y, seq_len=512, n_splits=5, test_size=int(0.2 * len(X))):
        """Hyperparameter walk-forward cross-validation specification."""
        # Isolating 5th fold away from hyperparameter tuning
        X_cv = X.iloc[:-test_size]
        y_cv = y.iloc[:-test_size]

        # Chronological splits with a purged gap to prevent sequence overlap leakage
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=seq_len)
        hit_rates = []

        for train_index, val_index in tscv.split(X_cv):
            # Using .iloc because X and y are now Pandas DataFrames/Series
            X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
            y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            eval_list = [(dtrain, 'train'), (dval, 'eval')]

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=eval_list,
                early_stopping_rounds=100,
                verbose_eval=False
            )

            # Predict out-of-sample to calculate metrics
            preds_continuous = model.predict(dval)

            # Convert continuous return into Directional Signs (-1 for DOWN, 1 for UP)
            true_direction = np.sign(y_val.values)
            pred_direction = np.sign(preds_continuous)

            # Calculate Directional Accuracy (Hit Rate) for this fold
            fold_hit_rate = np.mean((true_direction == pred_direction) & (true_direction != 0))
            hit_rates.append(fold_hit_rate)

        # Return the average Hit Rate across all validation folds
        return np.mean(hit_rates)

    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': 'cuda',  # To route XGBoost training sessions to GPU
        'tree_method': 'hist',
        'seed': 42,  # Ensures XGBoost tree building (subsampling) is exactly reproducible
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'alpha': trial.suggest_float('alpha', 0.1, 10,  log=True),
        'lambda': trial.suggest_float('lambda', 0.1, 10, log=True)
    }
    avg_hit_rate = walk_forward_cv(param, X, y, seq_len)


    return avg_hit_rate

