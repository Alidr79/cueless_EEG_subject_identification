import numpy as np
import os
import joblib
from tqdm import tqdm
from scipy.signal import welch

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.base import clone

import pandas as pd

# File paths
TRAIN_DATASET_PATH = '../train_dataset.npy'
VAL_DATASET_PATH   = '../val_dataset.npy'
TEST_DATASET_PATH  = '../test_dataset.npy'

MODEL_PATH = 'XGB_on_psd_optuna.pkl'  # Trained and optimized model path
LOG_FILE   = 'test_results_XGB_psd_optuna_3seeds.md'

# Constants
SFREQ = 250
PSD_N_FREQS = 30
SEEDS = [42, 1, 97]

# GPU check
def is_gpu_available():
    try:
        import xgboost
        return xgboost.__version__ >= '1.0' and hasattr(xgboost, 'DeviceQuantileDMatrix')
    except:
        return False

gpu_available = is_gpu_available()

# Load EEG datasets
def load_datasets(train_path, val_path, test_path):
    train = np.load(train_path, allow_pickle=True).item()
    val   = np.load(val_path,   allow_pickle=True).item()
    test  = np.load(test_path,  allow_pickle=True).item()
    return (train['eeg_data'], train['subject'],
            val['eeg_data'],   val['subject'],
            test['eeg_data'],  test['subject'])

# Extract PSD features
def extract_psd_features(eeg_data, sfreq=250, n_freqs=30):
    n_epochs, n_channels, _ = eeg_data.shape
    features = np.zeros((n_epochs, n_channels * n_freqs), dtype=np.float32)

    for i in tqdm(range(n_epochs), desc="Extracting PSD"):
        feats = []
        for ch in range(n_channels):
            signal = eeg_data[i, ch].astype(np.float64)
            freqs, psd = welch(signal, fs=sfreq, nperseg=sfreq)
            psd_resized = np.interp(
                np.linspace(freqs[0], freqs[-1], n_freqs), freqs, psd
            )
            feats.append(psd_resized)
        features[i] = np.hstack(feats)
    return features

# Evaluation
def evaluate_model(model, X, y):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    prec  = precision_score(y, preds, average='macro')
    rec   = recall_score(y, preds, average='macro')
    return acc, prec, rec

# Logging helper
def log_write(file, text):
    with open(file, 'a') as f:
        f.write(text)

# MAIN
print("Loading data and extracting PSD features...")
eeg_train, y_train, eeg_val, y_val, eeg_test, y_test = load_datasets(
    TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH
)

X_train = extract_psd_features(eeg_train, sfreq=SFREQ, n_freqs=PSD_N_FREQS)
X_val   = extract_psd_features(eeg_val,   sfreq=SFREQ, n_freqs=PSD_N_FREQS)
X_test  = extract_psd_features(eeg_test,  sfreq=SFREQ, n_freqs=PSD_N_FREQS)

# Load best model (already optimized via Optuna)
original_pipeline: Pipeline = joblib.load(MODEL_PATH)
original_model = original_pipeline.named_steps['model']

# Run over 3 seeds
results = []
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

for seed in SEEDS:
    pipeline_clone = clone(original_pipeline)
    model_clone = pipeline_clone.named_steps['model']
    model_clone.set_params(random_state=seed)

    print(f"\nTraining model with random seed {seed}...")
    pipeline_clone.fit(X_train, y_train)

    acc, prec, rec = evaluate_model(pipeline_clone, X_test, y_test)
    results.append((acc, prec, rec))

    log_write(LOG_FILE, f"Seed {seed}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}\n")

# Aggregate
accs, precs, recs = zip(*results)
mean_acc, std_acc = np.mean(accs), np.std(accs)
mean_prec, std_prec = np.mean(precs), np.std(precs)
mean_rec, std_rec = np.mean(recs), np.std(recs)

log_write(LOG_FILE, "\n**Mean and Std over 3 seeds**\n")
log_write(LOG_FILE, f"- Accuracy : {mean_acc:.4f} ± {std_acc:.4f}\n")
log_write(LOG_FILE, f"- Precision: {mean_prec:.4f} ± {std_prec:.4f}\n")
log_write(LOG_FILE, f"- Recall   : {mean_rec:.4f} ± {std_rec:.4f}\n")

print(f"\nDone. Results saved to `{LOG_FILE}`.")
