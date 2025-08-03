import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mne_features.feature_extraction import FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import os


MODEL_PATH = 'SVM_on_wavelet_optuna.pkl'  # Path to the best SVM model
RESULTS_LOG = 'test_results_SVM_wavelet_per_class_per_word.md' # Log file for results


def load_datasets(train_path, val_path, test_path):
    """Loads the training, validation, and test datasets from .npy files."""
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


def calculate_per_word_accuracy(y_true, y_pred, word_labels, word_names=None):
    """Calculates accuracy for each distinct word."""
    per_word_accuracy = {}
    

    unique_words = np.unique(word_labels)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for word_idx in unique_words:

        word_mask = word_labels == word_idx
        
        if np.sum(word_mask) == 0:
            continue


        word_true = y_true[word_mask]
        word_pred = y_pred[word_mask]
        

        correct_predictions = np.sum(word_true == word_pred)
        total_predictions = len(word_true)
        word_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        word_name = word_names[word_idx] if word_names and word_idx < len(word_names) else f'Word_{word_idx}'
        per_word_accuracy[word_name] = word_accuracy
    
    return per_word_accuracy

def calculate_per_class_accuracy(y_true, y_pred, n_classes):
    """Calculates accuracy for each class and returns the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    per_class_accuracy = {}

    class_names = [f'sub-{i+1:02d}' for i in range(n_classes)]  
        
    for i in range(n_classes):

        tp = cm[i, i]

        total_class_samples = cm[i, :].sum()
        
        if total_class_samples > 0:
            class_accuracy = tp / total_class_samples
        else:
            class_accuracy = 0.0  
            
        per_class_accuracy[class_names[i]] = class_accuracy
    
    return per_class_accuracy, cm

def evaluate_model(seed, C, gamma, eeg_train, y_train, eeg_test, y_test):
    """Creates, trains, and evaluates the SVM pipeline for a given seed."""
    pipe = Pipeline([
        ('fe', FeatureExtractor(sfreq=250, selected_funcs=['wavelet_coef_energy'])),
        ('scaler', StandardScaler()),
        ('model', SVC(C=C, gamma=gamma, kernel='rbf', random_state=seed))
    ])

    print(f"--- Evaluating for Seed: {seed} ---")
    print("Fitting model...")
    pipe.fit(eeg_train, y_train)
    
    print("Predicting on test set...")
    preds = pipe.predict(eeg_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)

    return acc, prec, rec, preds

def log_write(file, text):
    """Appends a line of text to the specified log file."""
    with open(file, 'a') as f:
        f.write(text + '\n')

if __name__ == "__main__":
    eeg_train, y_train, eeg_val, y_val, eeg_test, y_test = load_datasets(
        '../train_dataset.npy', '../val_dataset.npy', '../test_dataset.npy'
    )

    base_pipeline = joblib.load(MODEL_PATH)
    base_model = base_pipeline.named_steps['model']
    C = base_model.C
    gamma = base_model.gamma
    
    n_classes = len(np.unique(y_train))

    print("Loading metadata for per-word analysis...")
    metadata = pd.read_csv("../all_metadata.csv")
    test_indices = metadata[metadata['session_id'] == 5].index.values
    word_labels = metadata.loc[test_indices, "words"].values - 1  # Subtract 1 to make it 0-indexed
    word_names = ['backward', 'forward', 'left', 'right', 'stop']

    if len(y_test) != len(word_labels):
        raise ValueError(
            f"Mismatch between number of test labels ({len(y_test)}) and "
            f"word labels in metadata ({len(word_labels)}). "
            "Please check if 'test_dataset.npy' corresponds to 'session_id == 5'."
        )



    if os.path.exists(RESULTS_LOG):
        os.remove(RESULTS_LOG)
        
    log_write(RESULTS_LOG, "# SVM Test Results (Seed 42)\n")
    log_write(RESULTS_LOG, f"Model: SVM with RBF Kernel")
    log_write(RESULTS_LOG, f"Hyperparameters: C={C:.6f}, gamma={gamma:.6f}\n")


    seed = 42
    acc, prec, rec, preds = evaluate_model(seed, C, gamma, eeg_train, y_train, eeg_test, y_test)


    log_write(RESULTS_LOG, f"## Overall Performance (Seed {seed})\n")
    log_write(RESULTS_LOG, f"- **Accuracy:** {acc:.4f}")
    log_write(RESULTS_LOG, f"- **Macro Precision:** {prec:.4f}")
    log_write(RESULTS_LOG, f"- **Macro Recall:** {rec:.4f}\n")


    per_class_acc, confusion_mat = calculate_per_class_accuracy(y_test, preds, n_classes)
    log_write(RESULTS_LOG, "## Per-Class Accuracy\n")
    for class_name, class_acc in per_class_acc.items():
        log_write(RESULTS_LOG, f"- **{class_name}:** {class_acc:.4f}")


    per_word_acc = calculate_per_word_accuracy(y_test, preds, word_labels, word_names)
    log_write(RESULTS_LOG, "\n## Per-Word Accuracy\n")
    for word_name, word_acc in per_word_acc.items():
        log_write(RESULTS_LOG, f"- **{word_name}:** {word_acc:.4f}")


    log_write(RESULTS_LOG, "\n## Confusion Matrix\n")
    log_write(RESULTS_LOG, f"```\n{confusion_mat}\n```")

    print(f"\nEvaluation complete. Results written to: {RESULTS_LOG}")
