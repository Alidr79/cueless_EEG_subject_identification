import random
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import time
from datetime import datetime
import json

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mne_features.feature_extraction import FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score

import optuna
from optuna.trial import Trial


BASE_DATA_PATH = '../'
RESULTS_DIR = 'svm_wavelet_scalability_results'
OPTUNA_N_TRIALS = 50  
SEED = 42


def control_randomness(seed: int):
    """Function to control randomness for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

control_randomness(SEED)


def create_subset_datasets(n_subjects, base_path=BASE_DATA_PATH, results_dir=RESULTS_DIR):
    """
    Creates and saves training, validation, and test datasets containing
    only the first n_subjects.
    """

    eeg_data = np.load(os.path.join(base_path, "all_eeg.npy"))
    metadata = pd.read_csv(os.path.join(base_path, "all_metadata.csv"))


    subject_mask = metadata['subject_id'] <= n_subjects
    filtered_metadata = metadata[subject_mask].copy()
    
    filtered_metadata['subject_id_0_indexed'] = filtered_metadata['subject_id'] - 1

    datasets = {}
    for session_name, session_ids in [('train', [1, 2, 3]), ('val', [4]), ('test', [5])]:
        session_mask = filtered_metadata['session_id'].isin(session_ids)
        session_indices = filtered_metadata[session_mask].index.values
        
        eeg_subset = eeg_data[session_indices]
        labels_subset = filtered_metadata.loc[session_indices, 'subject_id_0_indexed'].values
        
        data_dict = {'eeg_data': eeg_subset, 'subject': labels_subset}
        
        path = os.path.join(results_dir, f'{session_name}_dataset_{n_subjects}subjects.npy')
        np.save(path, data_dict)
        datasets[session_name] = (path, len(eeg_subset))

    return datasets['train'], datasets['val'], datasets['test']

def load_data_from_path(path):
    """Loads EEG data and labels from a .npy file."""
    data_dict = np.load(path, allow_pickle=True).item()
    return data_dict['eeg_data'], data_dict['subject']


def objective(trial: Trial, eeg_train, y_train, eeg_val, y_val):
    """Defines the objective for Optuna hyperparameter search."""

    C = trial.suggest_loguniform('C', 1e-3, 1e2)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e2)


    pipe = Pipeline([
        ('fe', FeatureExtractor(sfreq=250, selected_funcs=['wavelet_coef_energy'])),
        ('scaler', StandardScaler()),
        ('model', SVC(C=C, gamma=gamma, kernel='rbf', random_state=SEED))
    ])


    pipe.fit(eeg_train, y_train)
    val_preds = pipe.predict(eeg_val)
    val_acc = accuracy_score(y_val, val_preds)

    return val_acc


def run_scalability_experiment():
    """Runs the complete SVM scalability experiment from 2 to 10 subjects."""
    
    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Starting SVM Scalability Analysis: Performance vs. Number of Subjects")
    print("=" * 70)
    
    for n_subjects in range(2, 11):  # Loop from 2 to 10 subjects
        
        print(f"\n{'='*30} EXPERIMENT: {n_subjects} SUBJECTS {'='*30}")
        start_time = time.time()
        

        print(f"Creating datasets for {n_subjects} subjects...")
        (train_path, n_train), (val_path, n_val), (test_path, n_test) = create_subset_datasets(n_subjects, results_dir=RESULTS_DIR)
        
        eeg_train, y_train = load_data_from_path(train_path)
        eeg_val, y_val = load_data_from_path(val_path)
        eeg_test, y_test = load_data_from_path(test_path)
        
        print(f"Dataset sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")
        

        print(f"Starting hyperparameter optimization with {OPTUNA_N_TRIALS} trials...")
        study_name = f"study_SVM_wavelet_{n_subjects}subjects"
        storage_path = f"sqlite:///{RESULTS_DIR}/{study_name}.db"
        
        study = optuna.create_study(
            direction="maximize",
            storage=storage_path,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        study.optimize(lambda trial: objective(trial, eeg_train, y_train, eeg_val, y_val), n_trials=OPTUNA_N_TRIALS)
        
        best_params = study.best_trial.params
        print(f"Best validation accuracy from Optuna: {study.best_trial.value:.4f}")
        print(f"Best parameters: {best_params}")
        

        print("Training final model with best parameters...")
        final_pipeline = Pipeline([
            ('fe', FeatureExtractor(sfreq=250, selected_funcs=['wavelet_coef_energy'])),
            ('scaler', StandardScaler()),
            ('model', SVC(**best_params, kernel='rbf', random_state=SEED))
        ])
        
        final_pipeline.fit(eeg_train, y_train)
        

        test_preds = final_pipeline.predict(eeg_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, average='macro', zero_division=0)
        test_recall = recall_score(y_test, test_preds, average='macro', zero_division=0)
        
        duration = time.time() - start_time
        

        result = {
            'n_subjects': n_subjects,
            'n_train_samples': n_train,
            'n_val_samples': n_val,
            'n_test_samples': n_test,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'best_hyperparams': best_params,
            'optuna_val_accuracy': study.best_trial.value,
            'duration_minutes': duration / 60,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"--- Results for {n_subjects} subjects ---")
        print(f"  - Test Accuracy: {test_acc:.4f}")
        print(f"  - Test Precision: {test_precision:.4f}")
        print(f"  - Test Recall: {test_recall:.4f}")
        print(f"  - Duration: {duration/60:.2f} minutes")
        

        with open(os.path.join(RESULTS_DIR, 'svm_scalability_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
    return results


def generate_markdown_report(results):
    """Generates a markdown report from the experiment results."""
    report_path = os.path.join(RESULTS_DIR, 'svm_scalability_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# SVM Scalability Analysis Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report analyzes how the SVM model performance scales with an increasing number of subjects (classes).\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Subjects | Train Samples | Test Samples | Test Acc | Test Prec | Test Rec | Optuna Val Acc | Duration (min) |\n")
        f.write("|----------|---------------|--------------|----------|-----------|----------|----------------|----------------|\n")
        
        for r in results:
            f.write(f"| {r['n_subjects']} | {r['n_train_samples']} | {r['n_test_samples']} | ")
            f.write(f"{r['test_accuracy']:.4f} | {r['test_precision']:.4f} | {r['test_recall']:.4f} | ")
            f.write(f"{r['optuna_val_accuracy']:.4f} | {r['duration_minutes']:.1f} |\n")
            
        f.write("\n## Detailed Results & Hyperparameters\n\n")
        
        for r in results:
            f.write(f"### {r['n_subjects']} Subjects\n")
            f.write(f"- **Test Accuracy:** {r['test_accuracy']:.4f}\n")
            f.write(f"- **Best Hyperparameters (C):** {r['best_hyperparams']['C']:.2e}\n")
            f.write(f"- **Best Hyperparameters (gamma):** {r['best_hyperparams']['gamma']:.2e}\n\n")

    print(f"\nMarkdown report generated: {report_path}")


if __name__ == "__main__":

    final_results = run_scalability_experiment()
    

    if final_results:
        generate_markdown_report(final_results)
    
    print("\n" + "="*70)
    print("SVM SCALABILITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"All results and reports are saved in the '{RESULTS_DIR}' directory.")
