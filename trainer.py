import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from feature_engineering import compute_features
from adversarial_lightgbm import train_adversarial_model, predict_robust

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

        macro_df = data_manager.get_macro_data(df)
        if macro_df.empty:
            macro_df = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

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

        X = np.array(daily_features)
        y = np.array(daily_targets)
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = y.reshape(-1)
        split = int(0.8 * len(X_flat))
        X_train, X_val = X_flat[:split], X_flat[split:]
        y_train, y_val = y_flat[:split], y_flat[split:]

        # Train adversarial model using keyword arguments (except first five positional)
        model, scaler = train_adversarial_model(
            X_train, y_train, X_val, y_val, config.LGB_PARAMS,
            epsilon=config.EPSILON,
            pgd_steps=config.PGD_STEPS,
            pgd_alpha=config.PGD_ALPHA,
            adversarial_train=config.ADVERSARIAL_TRAIN
        )

        last_features, etf_names, _ = compute_features(returns, macro_df, window=20)
        robust_pred = predict_robust(model, last_features, scaler,
                                     epsilon=config.EPSILON,
                                     pgd_steps=config.PGD_STEPS,
                                     pgd_alpha=config.PGD_ALPHA)

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

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/adversarial_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Adversarial Robustness Engine complete ===")

if __name__ == "__main__":
    main()
