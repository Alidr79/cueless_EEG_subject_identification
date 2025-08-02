import numpy as np
import os
import joblib
from tqdm import tqdm
from scipy.signal import welch

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

import optuna
from optuna.trial import Trial


TRAIN_DATASET_PATH = '../train_dataset.npy'
VAL_DATASET_PATH   = '../val_dataset.npy'
TEST_DATASET_PATH  = '../test_dataset.npy'

MODEL_SAVE_PATH    = 'SVM_on_psd_optuna.pkl'
OPTUNA_STORAGE     = "sqlite:///study_SVM_psd.db"
LOG_TO_FILE        = 'test_results_SVM_psd_optuna.md'
OPTUNA_N_TRIALS    = 50

SFREQ = 250
PSD_N_FREQS = 30  # number of frequency bins to keep



def load_datasets(train_path, val_path, test_path):
    train = np.load(train_path, allow_pickle=True).item()
    val   = np.load(val_path,   allow_pickle=True).item()
    test  = np.load(test_path,  allow_pickle=True).item()
    return (train['eeg_data'], train['subject'],
            val['eeg_data'],   val['subject'],
            test['eeg_data'],  test['subject'])



def extract_psd_features(eeg_data, sfreq=250, n_freqs=30):
    """
    eeg_data shape: (n_samples, n_channels, n_times)
    Returns shape:  (n_samples, n_channels * n_freqs)
    """
    n_epochs, n_channels, _ = eeg_data.shape
    features = np.zeros((n_epochs, n_channels * n_freqs), dtype=np.float32)

    for i in tqdm(range(n_epochs), desc="Extracting PSD"):
        feats = []
        for ch in range(n_channels):
            signal = eeg_data[i, ch].astype(np.float64)
            freqs, psd = welch(signal, fs=sfreq, nperseg=sfreq)
            # Truncate or interpolate to fixed-length frequency features
            psd_resized = np.interp(
                np.linspace(freqs[0], freqs[-1], n_freqs), freqs, psd
            )
            feats.append(psd_resized)
        features[i] = np.hstack(feats)
    return features



print("Loading and extracting PSD features...")
eeg_train, y_train, eeg_val, y_val, eeg_test, y_test = load_datasets(
    TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH
)

X_train = extract_psd_features(eeg_train, sfreq=SFREQ, n_freqs=PSD_N_FREQS)
X_val   = extract_psd_features(eeg_val,   sfreq=SFREQ, n_freqs=PSD_N_FREQS)
X_test  = extract_psd_features(eeg_test,  sfreq=SFREQ, n_freqs=PSD_N_FREQS)



def objective(trial: Trial):
    C     = trial.suggest_loguniform('C',     1e-3, 1e1)
    gamma = trial.suggest_loguniform('gamma', 1e-3, 1e1)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(C=C, gamma=gamma, kernel='rbf', random_state=42))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    return accuracy_score(y_val, preds)



def run_optuna_study(n_trials=OPTUNA_N_TRIALS):
    study = optuna.create_study(
        direction="maximize",
        storage=OPTUNA_STORAGE,
        study_name="SVM_PSD_Study",
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



final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', random_state=42))
])
print("Training final model...")
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


with open(LOG_TO_FILE, 'a') as f:
    f.write(f"Test Accuracy: {test_acc:.4f}, "
            f"Precision: {test_prec:.4f}, "
            f"Recall: {test_rec:.4f}\n")

print("All done.")
