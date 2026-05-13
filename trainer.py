import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from feature_engineering import compute_features
from adversarial_lightgbm import train_adversarial_model, predict_robust

def create_ranking_data(features, returns_next_day):
    """
    Features: (n_samples, n_etfs, n_features)
    Returns_next_day: (n_samples, n_etfs)
    We need to reshape to (n_samples * n_etfs, n_features) and create groups per time point.
    """
    n_days, n_etfs, n_feat = features.shape
    X = features.reshape(-1, n_feat)
    y = returns_next_day.reshape(-1)
    # group: each day is a query (all ETFs in that day)
    group = np.repeat(np.arange(n_days), n_etfs)
    return X, y, group

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Adversarial Engine) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.TRAIN_WINDOW + 60:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Get macro data
        macro_df = data_manager.get_macro_data(df)
        if macro_df.empty:
            macro_df = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        # Build daily features and targets
        daily_features = []
        daily_targets = []
        start_idx = max(0, len(returns) - config.TRAIN_WINDOW - 50)
        for i in range(start_idx, len(returns) - 1):
            window_returns = returns.iloc[:i+1]
            if len(window_returns) < 60:
                continue
            features, _, _ = compute_features(window_returns, macro_df, window=20)
            target = returns.iloc[i+1].values
            daily_features.append(features)
            daily_targets.append(target)

        if len(daily_features) < 50:
            print("  Not enough daily samples")
            continue

        # Convert to arrays
        X = np.array(daily_features)          # (T, n_etfs, n_feat)
        y = np.array(daily_targets)           # (T, n_etfs)
        # Train/validation split (last 50 days for validation)
        split = -min(50, len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Create ranking format
        X_train_flat, y_train_flat, group_train = create_ranking_data(X_train, y_train)
        X_val_flat, y_val_flat, group_val = create_ranking_data(X_val, y_val)

        # Train adversarial model
        model, scaler = train_adversarial_model(
            X_train_flat, y_train_flat, group_train,
            X_val_flat, y_val_flat, group_val,
            config.LGB_PARAMS,
            epsilon=config.EPSILON,
            pgd_steps=config.PGD_STEPS,
            pgd_alpha=config.PGD_ALPHA,
            adversarial_train=config.ADVERSARIAL_TRAIN
        )

        # Predict for the most recent day (latest feature vector)
        last_features, etf_names, _ = compute_features(returns, macro_df, window=20)
        robust_pred = predict_robust(model, last_features, scaler,
                                     epsilon=config.EPSILON,
                                     pgd_steps=config.PGD_STEPS,
                                     pgd_alpha=config.PGD_ALPHA)

        # Rank ETFs by robust prediction
        sorted_idx = np.argsort(robust_pred)[::-1]
        top_etfs = []
        full_scores = {}
        for i, idx in enumerate(sorted_idx):
            ticker = etf_names[idx]
            pred = robust_pred[idx]
            full_scores[ticker] = float(pred)
            if i < config.TOP_N:
                top_etfs.append({"ticker": ticker, "robust_pred_return": float(pred)})
        print(f"  Top 3 ETFs by robust return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/adversarial_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Adversarial Robustness Engine complete ===")

if __name__ == "__main__":
    main()
