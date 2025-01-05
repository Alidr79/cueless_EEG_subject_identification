import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
from mne_features.feature_extraction import FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from tqdm import tqdm
import optuna
from optuna.trial import Trial
import os

# Define paths
TRAIN_DATASET_PATH = '../train_dataset.npy'
VAL_DATASET_PATH = '../val_dataset.npy'
TEST_DATASET_PATH = '../test_dataset.npy'
MODEL_SAVE_PATH = 'SVM_on_wavelet_optuna.pkl'  # Updated model save path
OPTUNA_STORAGE = "sqlite:///study_SVM_wavelet.db"  # Updated Optuna study storage
log_to_file = 'test_results_SVM_wavelet_optuna.md'  # Updated log file
OPTUNA_N_TRIALS = 50

# Load datasets
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
    TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH
)

# Define the objective function for Optuna
def objective(trial: Trial):
    # Define the hyperparameter search space for SVM with RBF kernel
    C = trial.suggest_loguniform('C', 1e-3, 1e1)
    gamma = trial.suggest_loguniform('gamma', 1e-3, 1e1)

    # Create the pipeline with StandardScaler
    pipe_steps = [
        ('fe', FeatureExtractor(
            sfreq=250,
            selected_funcs=['wavelet_coef_energy']
        )),
        ('scaler', StandardScaler()),  # Add StandardScaler
    ]
    
    # Define the SVM model with RBF kernel
    model = SVC(
        C=C,
        gamma=gamma,
        kernel='rbf',
        random_state=42
    )

    pipe_steps.append(('model', model))
    pipe = Pipeline(pipe_steps)

    # Fit the model
    pipe.fit(eeg_train, y_train)

    # Predict on validation set
    val_preds = pipe.predict(eeg_val)

    # Calculate validation accuracy
    val_acc = accuracy_score(y_val, val_preds)

    return val_acc

# Configure and run the Optuna study
def run_optuna_study(n_trials=OPTUNA_N_TRIALS):
    study = optuna.create_study(
        direction="maximize",
        storage=OPTUNA_STORAGE,
        study_name="SVM_Study_Enhanced",  # Updated study name
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, timeout=None)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Validation Accuracy): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial

# Run the Optuna study
best_trial = run_optuna_study(n_trials=OPTUNA_N_TRIALS)

# Retrieve best hyperparameters
best_params = best_trial.params

# Create the best model pipeline with StandardScaler
def create_best_pipeline(params):
    pipe_steps = [
        ('fe', FeatureExtractor(
            sfreq=250,
            selected_funcs=['wavelet_coef_energy']
        )),
        ('scaler', StandardScaler()),  # Add StandardScaler
    ]

    best_model = SVC(
        C=params['C'],
        gamma=params['gamma'],
        kernel='rbf',
        random_state=42
    )

    pipe_steps.append(('model', best_model))
    pipe = Pipeline(pipe_steps)

    return pipe

best_pipeline = create_best_pipeline(best_params)

# Fit the best model on the training data
print("Training the best model with optimal hyperparameters...")
best_pipeline.fit(eeg_train, y_train)

# Save the best model
joblib.dump(best_pipeline, MODEL_SAVE_PATH)
print(f"Best model saved to {MODEL_SAVE_PATH}")

# Evaluate on the test set
def evaluate_model(model, test_data, test_labels):
    test_preds = model.predict(test_data)
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='macro')
    test_recall = recall_score(test_labels, test_preds, average='macro')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    return test_acc, test_precision, test_recall

test_acc, test_precision, test_recall = evaluate_model(best_pipeline, eeg_test, y_test)

def log_write(file, log_input):
    with open(file, 'a') as log_file:
        log_file.write(log_input)

log_write(log_to_file, f"Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}\n")

print("Optuna hyperparameter tuning and evaluation complete.")
