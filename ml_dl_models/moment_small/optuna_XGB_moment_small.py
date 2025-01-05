import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import optuna
from optuna.trial import Trial
from xgboost import XGBClassifier
import os

# Define paths
# EMBEDDINGS_PATH = '51-shot_small_embeddings.npy'
EMBEDDINGS_PATH = 'zero-shot_small_embeddings.npy'

METADATA_PATH = '../all_metadata.csv'
MODEL_SAVE_PATH = 'XGBoost_on_MOMENT_small_zero-shot_optuna.pkl'  # Updated model save path
OPTUNA_STORAGE = "sqlite:///study_XGBoost_moment_small_zero-shot.db"  # Updated Optuna study storage
LOG_FILE = 'test_results_XGBoost_on_MOMENT_small_zero-shot_optuna.md'  # Updated log file
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

# Determine if GPU is available
def is_gpu_available():
    try:
        from xgboost import XGBClassifier
        import xgboost
        return xgboost.__version__ >= '1.0' and xgboost.DeviceQuantileDMatrix
    except:
        return False

gpu_available = is_gpu_available()

# Define the objective function for Optuna
def objective(trial: Trial):
    # Define the hyperparameter search space for XGBoost
    n_estimators = trial.suggest_int('n_estimators', 5, 1000, step=20)
    max_depth = trial.suggest_int('max_depth', 2, 15)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
    subsample = trial.suggest_uniform('subsample', 0.25, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 10.0)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-4, 10.0)

    pipe_steps = []

    # Configure tree_method and predictor based on GPU availability
    if gpu_available:
        tree_method = 'gpu_hist'
        predictor = 'gpu_predictor'
    else:
        tree_method = 'hist'
        predictor = 'cpu_predictor'    

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=tree_method,  # Use GPU-based tree construction if available
        predictor=predictor,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        n_jobs=-1  # Utilize all available CPU cores
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
    study_name="XGBoost_Study_Embeddings",
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

pipe_steps = []

# Configure tree_method and predictor based on GPU availability
if gpu_available:
    tree_method = 'gpu_hist'
    predictor = 'gpu_predictor'
else:
    tree_method = 'hist'
    predictor = 'cpu_predictor' 

best_model = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        min_child_weight=best_params['min_child_weight'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        tree_method=tree_method,
        predictor=predictor,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        n_jobs=-1  # Utilize all available CPU cores
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