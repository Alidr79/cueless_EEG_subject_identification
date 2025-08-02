import numpy as np
import os
import joblib
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

import optuna
from optuna.trial import Trial

# === Config ===
TRAIN_DATASET_PATH = '../train_dataset.npy'
VAL_DATASET_PATH   = '../val_dataset.npy'
TEST_DATASET_PATH  = '../test_dataset.npy'

MODEL_SAVE_PATH    = 'XGB_on_AR_optuna.pkl'
OPTUNA_STORAGE     = "sqlite:///study_XGB_AR.db"
LOG_TO_FILE        = 'test_results_XGB_AR_optuna.md'
OPTUNA_N_TRIALS    = 50

AR_ORDER = 6


def load_datasets(train_path, val_path, test_path):
    train = np.load(train_path, allow_pickle=True).item()
    val   = np.load(val_path,   allow_pickle=True).item()
    test  = np.load(test_path,  allow_pickle=True).item()
    return (train['eeg_data'], train['subject'],
            val['eeg_data'],   val['subject'],
            test['eeg_data'],  test['subject'])


def extract_ar_features(eeg_data, ar_order=6):
    """
    eeg_data shape: (n_samples, n_channels, n_times)
    Returns shape:  (n_samples, n_channels * ar_order)
    """
    n_epochs, n_channels, _ = eeg_data.shape
    features = np.zeros((n_epochs, n_channels * ar_order), dtype=np.float32)

    for i in tqdm(range(n_epochs), desc="Extracting AR"):
        feats = []
        for ch in range(n_channels):
            signal = eeg_data[i, ch].astype(np.float64)
            try:
                model = AutoReg(signal, lags=ar_order, old_names=False).fit()
                coeffs = model.params[1:]  # exclude intercept
            except Exception:
                coeffs = np.zeros(ar_order)
            feats.append(coeffs)
        features[i] = np.hstack(feats)
    return features


def is_gpu_available():
    try:
        import xgboost
        return xgboost.__version__ >= '1.0' and hasattr(xgboost, 'DeviceQuantileDMatrix')
    except:
        return False

gpu_available = is_gpu_available()


print("Loading data and extracting AR features...")
eeg_train, y_train, eeg_val, y_val, eeg_test, y_test = load_datasets(
    TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH
)

X_train = extract_ar_features(eeg_train, ar_order=AR_ORDER)
X_val   = extract_ar_features(eeg_val,   ar_order=AR_ORDER)
X_test  = extract_ar_features(eeg_test,  ar_order=AR_ORDER)


def objective(trial: Trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    if gpu_available:
        params['tree_method'] = 'gpu_hist'
        params['predictor'] = 'gpu_predictor'
    else:
        params['tree_method'] = 'hist'
        params['predictor'] = 'cpu_predictor'

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(**params))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    return accuracy_score(y_val, preds)


def run_optuna_study(n_trials=OPTUNA_N_TRIALS):
    study = optuna.create_study(
        direction="maximize",
        storage=OPTUNA_STORAGE,
        study_name="XGB_AR_Study",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"Finished {len(study.trials)} trials.")
    print("Best trial:")
    bt = study.best_trial
    print(f"  Validation Accuracy = {bt.value:.4f}")
    for k, v in bt.params.items():
        print(f"    {k}: {v}")
    return bt

best_trial = run_optuna_study()
best_params = best_trial.params


def create_final_pipeline(params):
    if gpu_available:
        params['tree_method'] = 'gpu_hist'
        params['predictor'] = 'gpu_predictor'
    else:
        params['tree_method'] = 'hist'
        params['predictor'] = 'cpu_predictor'

    model = XGBClassifier(
        **params,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

final_model = create_final_pipeline(best_params)
print("Training final XGB model on full training set...")
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, MODEL_SAVE_PATH)
print(f"Saved model to {MODEL_SAVE_PATH}")


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average='macro')
    rec   = recall_score(y_test, preds, average='macro')
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall   : {rec:.4f}")
    return acc, prec, rec

test_acc, test_prec, test_rec = evaluate_model(final_model, X_test, y_test)

# Log results
with open(LOG_TO_FILE, 'a') as f:
    f.write(f"Test Accuracy: {test_acc:.4f}, "
            f"Precision: {test_prec:.4f}, "
            f"Recall: {test_rec:.4f}\n")

print("All done.")
