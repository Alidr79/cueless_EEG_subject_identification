import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import os

# Paths
EMBEDDINGS_PATH = '32-shot_base_embeddings.npy'
METADATA_PATH = '../all_metadata.csv'
MODEL_PATH = 'XGBoost_on_MOMENT_base_finetuned_optuna.pkl'
LOG_FILE = 'test_results_best_XGBoost_moment_base_3randomseeds.md'

# Load data
def load_datasets(embeddings_path, metadata_path):
    embeddings_all = np.load(embeddings_path)
    all_metadata = pd.read_csv(metadata_path)
    y_all = all_metadata['subject_id'].values - 1

    train_idx = all_metadata[all_metadata['session_id'].isin([1, 2, 3])].index.values
    test_idx = all_metadata[all_metadata['session_id'] == 5].index.values

    X_train, y_train = embeddings_all[train_idx], y_all[train_idx]
    X_test, y_test = embeddings_all[test_idx], y_all[test_idx]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_datasets(EMBEDDINGS_PATH, METADATA_PATH)

# Load base pipeline
original_pipeline: Pipeline = joblib.load(MODEL_PATH)
original_model = original_pipeline.named_steps['model']

# Seeds to test
seeds = [42, 1, 97]
results = []

for seed in seeds:
    # Clone pipeline and model
    pipeline_clone = clone(original_pipeline)
    model_clone = pipeline_clone.named_steps['model']

    # Set new random_state
    model_clone.set_params(random_state=seed)


    print("current model:", pipeline_clone.named_steps['model'])
    print("With random seed:", pipeline_clone.named_steps['model'].get_params()['random_state'])
    # Fit with same training data
    pipeline_clone.fit(X_train, y_train)

    # Predict and evaluate
    preds = pipeline_clone.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')

    results.append((acc, prec, rec))

# Aggregate results
accs, precs, recs = zip(*results)
mean_acc, std_acc = np.mean(accs), np.std(accs)
mean_prec, std_prec = np.mean(precs), np.std(precs)
mean_rec, std_rec = np.mean(recs), np.std(recs)

# Write results to log
def log_write(file, text):
    with open(file, 'a') as f:
        f.write(text)

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

for i, (acc, prec, rec) in enumerate(results):
    log_write(LOG_FILE, f"Seed {seeds[i]}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}\n")

log_write(LOG_FILE, "\n**Mean and Std over 3 seeds**\n")
log_write(LOG_FILE, f"- Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
log_write(LOG_FILE, f"- Precision: {mean_prec:.4f} ± {std_prec:.4f}\n")
log_write(LOG_FILE, f"- Recall: {mean_rec:.4f} ± {std_rec:.4f}\n")

print(f"Done. Results saved to `{LOG_FILE}`.")