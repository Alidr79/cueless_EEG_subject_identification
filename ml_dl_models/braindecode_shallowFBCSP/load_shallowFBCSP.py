import pandas as pd
from tqdm import tqdm
import numpy as np
import os 
import random
import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
import optuna
from optuna.trial import Trial
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix  # Import metrics


def log_write(file, log_input):
    with open(file, 'a') as log_file:
        log_file.write(log_input)

SEED = 42
# SEED = 1
# SEED = 97


CHECKPOINT_LOAD_PATH = f"seed_{SEED}_checkpoints_ShallowFBCSPNet_train-1,2,3_val-4_test-5"
# CHECKPOINT_LOAD_PATH = f"checkpoints_ShallowFBCSPNet_train-3_val-4_test-5"
checkpoint_path = CHECKPOINT_LOAD_PATH + "/model_epoch_best_12.pth"

log_to_file = f'per_class_and_per_word_test_results_ShallowFBCSPNet_train-1,2,3_val-4_test-5.md'
log_write(log_to_file, f"train on ses-1,2,3 val on ses-4 and test on ses-5")
# log_to_file = f'seed_{SEED}_test_results_ShallowFBCSPNet_train-1,2,3_val-4_test-5.md'
# log_write(log_to_file, f"train on ses-1,2,3 val on ses-4 and test on ses-5\nseed_{SEED}\n")
# log_to_file = f'test_results_ShallowFBCSPNet_train-3_val-4_test-5.md'
# log_write(log_to_file, f"train on ses-3 val on ses-4 and test on ses-5\n")


# For more consistency across GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def control_randomness(seed: int = 42):
    """Function to control randomness in the code."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Control randomness
control_randomness(SEED)
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
seed = SEED
set_random_seeds(seed=seed, cuda=cuda)

# For dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

n_classes = 11
classes = list(range(n_classes))
n_channels = 30
input_window_samples = 500

model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto")

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure device is set correctly
model.to(device)  # Move model to the appropriate device

checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure the checkpoint is loaded on the correct device
model.load_state_dict(checkpoint['model_state_dict'])

# Define loss function (assuming CrossEntropyLoss)
loss_fn = torch.nn.CrossEntropyLoss()

def calculate_per_word_accuracy(y_true, y_pred, word_labels, word_names=None):
    
    per_word_accuracy = {}
        
    # Get unique words
    unique_words = np.unique(word_labels)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for word_idx in unique_words:
        # Find all samples for this word
        word_mask = word_labels == word_idx
        
        if np.sum(word_mask) == 0:
            continue

        # Get true and predicted labels for this word only
        word_true = y_true[word_mask]
        word_pred = y_pred[word_mask]
        
        # Calculate accuracy for this word
        correct_predictions = np.sum(word_true == word_pred)
        total_predictions = len(word_true)
        word_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        word_name = word_names[word_idx] if word_idx < len(word_names) else f'Word_{word_idx}'
        per_word_accuracy[word_name] = word_accuracy
    
    return per_word_accuracy

def calculate_per_class_accuracy(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    per_class_accuracy = {}
    class_names = [f'sub-{i+1:02d}' for i in range(n_classes)] 
        
    for i in range(n_classes):
        # True Positives for class i
        tp = cm[i, i]
        # Total samples for class i
        total_class_samples = cm[i, :].sum()
        
        if total_class_samples > 0:
            class_accuracy = tp / total_class_samples
        else:
            class_accuracy = 0.0  # No samples for this class
            
        per_class_accuracy[class_names[i]] = class_accuracy
    
    return per_class_accuracy, cm


@torch.no_grad()
def test_model(
    dataloader: DataLoader, model: Module, loss_fn, word_labels=None, word_names=None, print_batch_stats=True, phase = "test"
):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()  # Switch to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []

    if print_batch_stats:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    for batch_idx, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        batch_loss = loss_fn(pred, y).item()

        total_loss += batch_loss

        # Get the predicted class
        preds = torch.argmax(pred, dim=1)

        # Collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        if print_batch_stats:
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {batch_loss:.6f}"
            )

    average_loss = total_loss / n_batches
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    per_class_acc, confusion_mat = calculate_per_class_accuracy(all_labels, all_preds, n_classes)

    print(f"{phase} Accuracy: {100 * accuracy:.2f}%")
    print(f"{phase} Macro Precision: {100 * macro_precision:.2f}%")
    print(f"{phase} Macro Recall: {100 * macro_recall:.2f}%")
    print(f"{phase} Loss: {average_loss:.6f}\n")
    
    print("Per-Class Accuracies:")
    for class_name, acc in per_class_acc.items():
        print(f"{class_name}: {100 * acc:.2f}%")
    print()
    
    print("Confusion Matrix:")
    print(confusion_mat)
    print()

  # Calculate and print per-word accuracies if word labels are provided
    per_word_acc = None
    if word_labels is not None:
        per_word_acc = calculate_per_word_accuracy(all_labels, all_preds, word_labels, word_names)
        
        print("Per-Word Accuracies:")
        for word_name, acc in per_word_acc.items():
            print(f"{word_name}: {100 * acc:.2f}%")
        print()

    return average_loss, accuracy, macro_precision, macro_recall, per_class_acc, confusion_mat, per_word_acc 



class NpyDataset(Dataset):
    def __init__(self, npy_file):
        # Load the .npy file as a dictionary
        data_dict = np.load(npy_file, allow_pickle=True).item()
        self.data = data_dict['eeg_data']
        self.labels = data_dict['subject']
    
    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a sample and its corresponding label
        sample = self.data[idx]
        label = self.labels[idx]

        # Convert the padded sample to a torch tensor
        sample = torch.tensor(sample, dtype=torch.float)
        
        # Convert the label to a tensor (optional: depending on its type)
        label = torch.tensor(label, dtype=torch.long)  # Assuming labels are integer class indices
        
        # Return both the padded sample and the label
        return sample, label



test_dataset = NpyDataset('../test_dataset.npy')
test_loader  = DataLoader(test_dataset,  batch_size=16,
                              shuffle=False, num_workers=2,
                              worker_init_fn=seed_worker, generator=g)

metadata = pd.read_csv("../all_metadata.csv")
test_indices = metadata[metadata['session_id'] == 5].index.values
word_labels = metadata.loc[test_indices, "words"].values - 1  ## start from 0

word_names = ['backward', 'forward', 'left', 'right', 'stop']



test_loss, test_accuracy, test_macro_precision, test_macro_recall, per_class_accuracies, confusion_matrix_result, per_word_accuracies = test_model(test_loader, model, loss_fn, word_labels=word_labels, word_names=word_names)


log_write(log_to_file, f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_macro_precision:.4f}, Test Recall: {test_macro_recall:.4f}\n")

log_write(log_to_file, "Per-Class Accuracies:\n")
for class_name, acc in per_class_accuracies.items():
    log_write(log_to_file, f"{class_name}: {acc:.4f}\n")

log_write(log_to_file, f"Confusion Matrix:\n{confusion_matrix_result}\n")

# Log per-word accuracies if available
if per_word_accuracies is not None:
    log_write(log_to_file, "\n\nPer-Word Accuracies:\n")
    for word_name, acc in per_word_accuracies.items():
        log_write(log_to_file, f"{word_name}: {acc:.4f}\n")

print(test_accuracy, test_macro_precision, test_macro_recall)
