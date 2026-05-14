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
        if returns.empty or len(returns) < max(config.WINDOWS) + 60:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": [], "full_results": {}}
            continue

        macro_df = data_manager.get_macro_data(df)
        if macro_df.empty:
            macro_df = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}   # ticker -> (best_pred, best_window)
        window_results = {}  # store predictions per window

        for win in config.WINDOWS:
            print(f"  Training on window {win} days...")
            if len(returns) < win + 60:
                print(f"    Skipping (insufficient data)")
                continue

            daily_features = []
            daily_targets = []
            start_idx = max(0, len(returns) - win - 50)
            for i in range(start_idx, len(returns) - 1):
                window_returns = returns.iloc[:i+1]
                if len(window_returns) < 60:
                    continue
                features, _, _ = compute_features(window_returns, macro_df, window=20)
                target = returns.iloc[i+1].values
                daily_features.append(features)
                daily_targets.append(target)

            if len(daily_features) < 50:
                print(f"    Not enough daily samples for window {win}, skipping")
                continue

            X = np.array(daily_features)          # (T, n_etfs, n_feat)
            y = np.array(daily_targets)           # (T, n_etfs)
            X_flat = X.reshape(-1, X.shape[-1])
            y_flat = y.reshape(-1)
            split = int(0.8 * len(X_flat))
            X_train, X_val = X_flat[:split], X_flat[split:]
            y_train, y_val = y_flat[:split], y_flat[split:]

            model, scaler = train_adversarial_model(
                X_train, y_train, X_val, y_val,
                config.LGB_PARAMS,
                epsilon=config.EPSILON,
                pgd_steps=config.PGD_STEPS,
                pgd_alpha=config.PGD_ALPHA,
                adversarial_train=config.ADVERSARIAL_TRAIN
            )

            # Predict for the most recent day
            last_features, etf_names, _ = compute_features(returns, macro_df, window=20)
            robust_pred = predict_robust(model, last_features, scaler,
                                         epsilon=config.EPSILON,
                                         pgd_steps=config.PGD_STEPS,
                                         pgd_alpha=config.PGD_ALPHA)

            # Store predictions for this window
            for idx, ticker in enumerate(etf_names):
                pred = robust_pred[idx]
                if ticker not in best_per_etf or pred > best_per_etf[ticker][0]:
                    best_per_etf[ticker] = (pred, win)

            window_results[f"window_{win}"] = {
                "predictions": {ticker: float(robust_pred[i]) for i, ticker in enumerate(etf_names)},
                "top_etfs": [{"ticker": etf_names[i], "robust_pred_return": float(robust_pred[i])}
                             for i in np.argsort(robust_pred)[::-1][:config.TOP_N]]
            }
            print(f"    Completed window {win}")

        # Build final results: top 3 ETFs based on best prediction across windows
        if not best_per_etf:
            print("  No valid windows")
            all_results[universe_name] = {"top_etfs": [], "full_results": {}}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (pred, win) in sorted_etfs:
            full_scores[ticker] = {"best_pred_return": pred, "best_window": win}
            if len(top_etfs) < config.TOP_N:
                top_etfs.append({"ticker": ticker, "robust_pred_return": pred, "best_window": win})

        print(f"  Top 3 ETFs (best across windows):")
        for etf in top_etfs:
            print(f"    {etf['ticker']}: pred={etf['robust_pred_return']:.4f} (window={etf['best_window']}d)")

        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/adversarial_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Adversarial Robustness Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
