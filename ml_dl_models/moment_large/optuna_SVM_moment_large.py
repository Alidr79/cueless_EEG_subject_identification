import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import optuna
from optuna.trial import Trial
from itertools import product
from tqdm import tqdm
import os

# Define paths
# EMBEDDINGS_PATH = '35-shot_large_embeddings.npy'
EMBEDDINGS_PATH = 'zero-shot_large_embeddings.npy'

METADATA_PATH = '../all_metadata.csv'
MODEL_SAVE_PATH = 'SVM_on_MOMENT_large_zero-shot_optuna.pkl'  # Updated model save path
OPTUNA_STORAGE = "sqlite:///study_SVM_moment_large_zero-shot.db"  # Updated Optuna study storage
LOG_FILE = 'test_results_SVM_on_MOMENT_large_zero-shot_optuna.md'  # Updated log file
OPTUNA_N_TRIALS = 50

# Function to log messages
def log_write(file, log_input):
    with open(file, 'a') as log_file:
        log_file.write(log_input)

# Load datasets
def load_datasets(embeddings_path, metadata_path):
    embeddings_all = np.load(embeddings_path)
    all_metadata = pd.read_csv(metadata_path)
    y_all = all_metadata['subject_id'].values - 1  # Assuming subject IDs start from 1

    # Define train, validation, and test indices based on session_id
    train_indices = all_metadata[all_metadata['session_id'].isin([1, 2, 3])].index.values
    val_indices = all_metadata[all_metadata['session_id'] == 4].index.values
    test_indices = all_metadata[all_metadata['session_id'] == 5].index.values

    embeddings_train = embeddings_all[train_indices]
    embeddings_val = embeddings_all[val_indices]
    embeddings_test = embeddings_all[test_indices]

    y_train = y_all[train_indices]
    y_val = y_all[val_indices]
    y_test = y_all[test_indices]

    return embeddings_train, y_train, embeddings_val, y_val, embeddings_test, y_test

embeddings_train, y_train, embeddings_val, y_val, embeddings_test, y_test = load_datasets(
    EMBEDDINGS_PATH, METADATA_PATH
)

# Define the objective function for Optuna
def objective(trial: Trial):
    # Define the hyperparameter search space for SVM with RBF kernel
    C = trial.suggest_loguniform('C', 1e-3, 1e1)
    gamma = trial.suggest_loguniform('gamma', 1e-3, 1e1)

    # Create the pipeline with optional StandardScaler
    pipe_steps = []
    
    pipe_steps.append(('scaler', StandardScaler()))
    
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
    pipe.fit(embeddings_train, y_train)

    # Predict on validation set
    val_preds = pipe.predict(embeddings_val)

    # Calculate validation accuracy
    val_acc = accuracy_score(y_val, val_preds)

    return val_acc

# Configure and run the Optuna study
study = optuna.create_study(
    direction="maximize",
    storage=OPTUNA_STORAGE,
    study_name="SVM_Study_Embeddings",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)
study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=None)

print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
best_trial = study.best_trial

print(f"  Value (Validation Accuracy): {best_trial.value:.4f}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Retrieve best hyperparameters
best_params = best_trial.params

# Create the best model pipeline with optional scaler
pipe_steps = []

pipe_steps.append(('scaler', StandardScaler()))

best_model = SVC(
    C=best_params['C'],
    gamma=best_params['gamma'],
    kernel='rbf',
    random_state=42
)

pipe_steps.append(('model', best_model))
best_pipeline = Pipeline(pipe_steps)

# Fit the best model on the training data
print("Training the best model with optimal hyperparameters...")
best_pipeline.fit(embeddings_train, y_train)

# Save the best model
joblib.dump(best_pipeline, MODEL_SAVE_PATH)
print(f"Best model saved to {MODEL_SAVE_PATH}")

# Evaluate on the test set
test_preds = best_pipeline.predict(embeddings_test)
test_acc = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='macro')
test_recall = recall_score(y_test, test_preds, average='macro')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Log the test results
# Remove existing log file if it exists
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

log_write(LOG_FILE, f"Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}\n")
log_write(LOG_FILE, f"Best Hyperparameters: {best_params}\n")

print("Optuna hyperparameter tuning and evaluation complete.")
