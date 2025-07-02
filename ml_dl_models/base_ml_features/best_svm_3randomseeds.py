import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mne_features.feature_extraction import FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

MODEL_PATH = 'SVM_on_wavelet_optuna.pkl'  # Best SVM model
RESULTS_LOG = 'test_results_SVM_wavelet_3seeds.md'

def load_datasets(train_path, val_path, test_path):
    train_dataset = np.load(train_path, allow_pickle=True).item()
    val_dataset = np.load(val_path, allow_pickle=True).item()
    test_dataset = np.load(test_path, allow_pickle=True).item()

    eeg_train = train_dataset['eeg_data']
    eeg_val = val_dataset['eeg_data']
    eeg_test = test_dataset['eeg_data']
    y_train = train_dataset['subject']
    y_val = val_dataset['subject']
    y_test = test_dataset['subject']

    return eeg_train, y_train, eeg_val, y_val, eeg_test, y_test

eeg_train, y_train, eeg_val, y_val, eeg_test, y_test = load_datasets(
    '../train_dataset.npy', '../val_dataset.npy', '../test_dataset.npy'
)


base_pipeline = joblib.load(MODEL_PATH)
base_model = base_pipeline.named_steps['model']

C = base_model.C
gamma = base_model.gamma

# Evaluation metrics storage
accuracies, precisions, recalls = [], [], []

# Seeds to evaluate
seeds = [42, 1, 97]

def evaluate_model(seed):
    pipe = Pipeline([
        ('fe', FeatureExtractor(sfreq=250, selected_funcs=['wavelet_coef_energy'])),
        ('scaler', StandardScaler()),
        ('model', SVC(C=C, gamma=gamma, kernel='rbf', random_state=seed))
    ])

    print("current model:", pipe.named_steps['model'])

    pipe.fit(eeg_train, y_train)
    preds = pipe.predict(eeg_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')

    return acc, prec, rec

# Logging helper
def log_write(file, text):
    with open(file, 'a') as f:
        f.write(text + '\n')

# Start logging
log_write(RESULTS_LOG, "# SVM Test Results with 3 Seeds\n")
log_write(RESULTS_LOG, f"Fixed Hyperparameters: C={C:.6f}, gamma={gamma:.6f}\n")

for seed in seeds:
    acc, prec, rec = evaluate_model(seed)
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

    log_write(RESULTS_LOG, f"**Seed {seed}** → Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# Compute mean and std
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
mean_prec = np.mean(precisions)
std_prec = np.std(precisions)
mean_rec = np.mean(recalls)
std_rec = np.std(recalls)

log_write(RESULTS_LOG, "\n**Overall Performance**")
log_write(RESULTS_LOG, f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
log_write(RESULTS_LOG, f"Precision: {mean_prec:.4f} ± {std_prec:.4f}")
log_write(RESULTS_LOG, f"Recall: {mean_rec:.4f} ± {std_rec:.4f}")

print("Evaluation complete. Results written to:", RESULTS_LOG)
