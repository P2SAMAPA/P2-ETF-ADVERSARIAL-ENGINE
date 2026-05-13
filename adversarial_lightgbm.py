import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

def fgsm_attack(model, X, y, epsilon, scaler):
    """FGSM attack for regression."""
    X_adv = X.copy()
    for i in range(X.shape[0]):
        pred_orig = model.predict(X[i:i+1])[0]
        grad = np.zeros_like(X[i])
        for j in range(X[i].shape[0]):
            X_plus = X[i].copy()
            X_plus[j] += 1e-5
            pred_plus = model.predict(X_plus.reshape(1,-1))[0]
            grad[j] = (pred_plus - pred_orig) / 1e-5
        X_adv[i] = X[i] + epsilon * np.sign(grad)
    X_adv = np.clip(X_adv, X - 3*scaler.scale_, X + 3*scaler.scale_)
    return X_adv

def pgd_attack(model, X, y, epsilon, steps, alpha, scaler):
    X_adv = X.copy()
    for _ in range(steps):
        grad = np.zeros_like(X_adv)
        for i in range(X_adv.shape[0]):
            pred_orig = model.predict(X_adv[i:i+1])[0]
            for j in range(X_adv.shape[1]):
                X_plus = X_adv[i].copy()
                X_plus[j] += 1e-5
                pred_plus = model.predict(X_plus.reshape(1,-1))[0]
                grad[i,j] = (pred_plus - pred_orig) / 1e-5
        X_adv += alpha * np.sign(grad)
        perturbation = X_adv - X
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        X_adv = X + perturbation
    return X_adv

def train_adversarial_model(X_train, y_train, X_val, y_val, params, epsilon, pgd_steps, pgd_alpha, adversarial_train=True):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    valid_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
    model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=100, callbacks=[lgb.early_stopping(10)])

    if adversarial_train:
        # Generate adversarial examples on training set
        X_adv = pgd_attack(model, X_train_scaled, y_train, epsilon, pgd_steps, pgd_alpha, scaler)
        X_combined = np.vstack([X_train_scaled, X_adv])
        y_combined = np.hstack([y_train, y_train])
        adv_train_data = lgb.Dataset(X_combined, label=y_combined)
        model_adv = lgb.train(params, adv_train_data, valid_sets=[valid_data], num_boost_round=100, callbacks=[lgb.early_stopping(10)])
        return model_adv, scaler
    else:
        return model, scaler

def predict_robust(model, X, scaler, epsilon=0.5, pgd_steps=5, pgd_alpha=0.1):
    X_scaled = scaler.transform(X)
    X_adv = pgd_attack(model, X_scaled, None, epsilon, pgd_steps, pgd_alpha, scaler)
    pred_adv = model.predict(X_adv)
    return pred_adv
