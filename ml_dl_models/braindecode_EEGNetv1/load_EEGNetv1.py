import pandas as pd
from tqdm import tqdm
import numpy as np
import os 
import random
import torch
from braindecode.models import EEGNetv1
from braindecode.util import set_random_seeds
import optuna
from optuna.trial import Trial
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import precision_score, recall_score, accuracy_score  # Import metrics


def log_write(file, log_input):
    with open(file, 'a') as log_file:
        log_file.write(log_input)


CHECKPOINT_LOAD_PATH = "checkpoints_EEGNetv1_train-1,2,3_val-4_test-5"
checkpoint_path = CHECKPOINT_LOAD_PATH + "/model_epoch_best_8.pth"

log_to_file = 'test_results_EEGNetv1_train-1,2,3_val-4_test-5.md'
log_write(log_to_file, f"train on ses-1,2,3 val on ses-4 and test on ses-5\n")





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
control_randomness(42)
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
seed = 42
set_random_seeds(seed=seed, cuda=cuda)

# For dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

n_classes = 11
classes = list(range(n_classes))
n_channels = 30
input_window_samples = 500

model = EEGNetv1(
    n_chans = n_channels,
    n_outputs = n_classes,
    n_times = input_window_samples,
    sfreq = 250,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure device is set correctly
model.to(device)  # Move model to the appropriate device

checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure the checkpoint is loaded on the correct device
model.load_state_dict(checkpoint['model_state_dict'])

# Define loss function (assuming CrossEntropyLoss)
loss_fn = torch.nn.CrossEntropyLoss()

@torch.no_grad()
def test_model(
    dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True, phase = "test"
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

    print(f"{phase} Accuracy: {100 * accuracy:.2f}%")
    print(f"{phase} Macro Precision: {100 * macro_precision:.2f}%")
    print(f"{phase} Macro Recall: {100 * macro_recall:.2f}%")
    print(f"{phase} Loss: {average_loss:.6f}\n")

    return average_loss, accuracy, macro_precision, macro_recall



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

test_loss, test_accuracy, test_macro_precision, test_macro_recall = test_model(test_loader, model, loss_fn)


log_write(log_to_file, f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_macro_precision:.4f}, Test Recall: {test_macro_recall:.4f}\n")
print(test_accuracy, test_macro_precision, test_macro_recall)
