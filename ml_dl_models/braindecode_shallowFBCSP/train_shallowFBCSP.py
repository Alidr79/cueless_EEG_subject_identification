import random
import os
import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
import optuna
from optuna.trial import Trial
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler


# SEED = 42
# SEED = 1
SEED = 97


TRAIN_DATASET_PATH = '../train_dataset.npy'
CHECKPOINT_SAVE_PATH = f"seed_{SEED}_checkpoints_ShallowFBCSPNet_train-1,2,3_val-4_test-5"


# For more consistency across GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def control_randomness(seed: int):
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

######################################################
# Dataset and Dataloader
######################################################
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

        # Convert the sample to a torch tensor
        sample = torch.tensor(sample, dtype=torch.float)
        
        # Convert the label to a tensor (assuming labels are integer class indices)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label

# Create Datasets
train_dataset = NpyDataset(TRAIN_DATASET_PATH)
val_dataset = NpyDataset('../val_dataset.npy')
test_dataset = NpyDataset('../test_dataset.npy')

print("train data len:", len(train_dataset))
print("val data len:", len(val_dataset))
print("test data len:", len(test_dataset))

# Create DataLoaders
def get_dataloaders(train_batch_size, val_batch_size, test_batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=8,
                              worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=val_batch_size,
                              shuffle=False, num_workers=2,
                              worker_init_fn=seed_worker, generator=g)
    test_loader  = DataLoader(test_dataset,  batch_size=test_batch_size,
                              shuffle=False, num_workers=2,
                              worker_init_fn=seed_worker, generator=g)
    return train_loader, val_loader, test_loader


######################################################
# Training and Evaluation
######################################################
def train_one_epoch(
        dataloader: DataLoader, model: Module, loss_fn, optimizer,
        scheduler: LRScheduler, epoch: int, device, print_batch_stats=True
):
    model.train()  # Set the model to training mode
    train_loss, correct = 0, 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        disable=not print_batch_stats)

    for batch_idx, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
            )

    # Update the learning rate
    scheduler.step()

    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model(
    dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True, phase="val"
):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()  # Switch to evaluation mode
    test_loss, correct = 0, 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader)) if print_batch_stats else enumerate(dataloader)

    for batch_idx, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        batch_loss = loss_fn(pred, y).item()

        test_loss += batch_loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"{phase} Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {batch_loss:.6f}"
            )

    test_loss /= n_batches
    correct /= size

    print(f"{phase} Accuracy: {100 * correct:.1f}%, {phase} Loss: {test_loss:.6f}\n")
    return test_loss, correct


def save_model(model, optimizer, scheduler, epoch, save_path = CHECKPOINT_SAVE_PATH, extra_file_str=""):
    os.makedirs(save_path, exist_ok=True)  # Create directory if not exist
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    # Save the checkpoint
    save_file_path = os.path.join(save_path, f'model_epoch_{extra_file_str}{epoch}.pth')
    torch.save(checkpoint, save_file_path)
    print(f"Model saved to {save_file_path}")

def log_write(file, log_input):
    with open(file, 'a') as log_file:
        log_file.write(log_input)


######################################################
# Train model with fixed hyper-parameters
######################################################
if __name__ == "__main__":
    # Use fixed hyperparameters which are found by the optuna
    lr = 0.0016837527178224338
    weight_decay = 0.0006147975234358214
    train_batch_size = 32
    val_batch_size = 32

    # Build model
    model = ShallowFBCSPNet(
        n_channels,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto").to(device)

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_batch_size, val_batch_size)

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training with early stopping
    best_val_accuracy = 0.0
    patience = 6
    counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device
        )
        val_loss, val_accuracy = test_model(val_loader, model, loss_fn, phase="val")
        test_loss, test_accuracy = test_model(test_loader, model, loss_fn, phase="test")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, optimizer, scheduler, epoch, extra_file_str="best_")
            print(f"New best val accuracy: {val_accuracy:.4f} at epoch {epoch}")
            print("Test Accuracy:", test_accuracy)
            counter = 0
        else:
            counter += 1

        if counter > patience:
            print(f"Early stopping at epoch {epoch}. No val accuracy improvement for {patience} epochs.")
            save_model(model, optimizer, scheduler, epoch, extra_file_str="earlyStop_")
            break

    print("Training complete with fixed hyperparameters.")