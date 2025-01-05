import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


predictions = np.load('32-shot_base_labels.npy')


all_metadata = pd.read_csv('../all_metadata.csv')
y_all = all_metadata['subject_id'].values - 1


train_indices = all_metadata[all_metadata['session_id'].isin([1,2,3])].index.values
val_indices = all_metadata[all_metadata['session_id'] == 4].index.values
test_indices = all_metadata[all_metadata['session_id'] == 5].index.values

y_train = y_all[train_indices]
y_val = y_all[val_indices]
y_test = y_all[test_indices]


pred_train = predictions[train_indices]
pred_val = predictions[val_indices]
pred_test = predictions[test_indices]

test_acc = accuracy_score(y_test, pred_test)
test_precision = precision_score(y_test, pred_test, average='macro')
test_recall = recall_score(y_test, pred_test, average='macro')


print("test acc = ", test_acc)
print("test precision = ", test_precision)
print("test recall = ", test_recall)


with open("MOMENT_base_full_finetuned_head_accuracy_report.md", "a") as log_file:  # Open in append mode
    log_file.write("32-shot_small_labels.npy")
    log_file.write(f"\n## Head classifier results : Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}\n")
