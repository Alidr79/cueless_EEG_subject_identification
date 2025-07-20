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


TRAIN_DATASET_PATH = '../dataset_ses-3.npy'
CHECKPOINT_SAVE_PATH = "checkpoints_ShallowFBCSPNet_train-3_val-4_test-5"
OPTUNA_STUDY_NAME = "study_ShallowFBCSPNet_train-3_val-4_test-5"
OPTUNA_SAVE_LOG = "sqlite:///study_ShallowFBCSPNet_train-3_val-4_test-5.sqlite3"
OPTUNA_N_TRIALS = 30


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
# Objective function for Optuna
######################################################
def objective(trial: Trial):
    """Optuna objective function to search for the best hyperparameters."""
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-2)
    train_batch_size = trial.suggest_categorical("train_batch_size", [16, 32, 64, 128])
    val_batch_size = trial.suggest_categorical("val_batch_size", [16, 32])

    # Create model with the fixed parameters from your code, or you can also search over them
    model = ShallowFBCSPNet(
        n_channels,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto").to(device)

    # Create Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(train_batch_size, val_batch_size)

    # Define optimizer, scheduler, loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # We can keep some best val accuracy tracking (or best val loss)
    best_val_accuracy = 0.0
    patience = 6
    counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device, print_batch_stats=False
        )
        val_loss, val_accuracy = test_model(val_loader, model, loss_fn, print_batch_stats=False, phase="val")
        # If you want to reduce time during hyperparam search, you can skip testing on test_loader for now

        # If val accuracy improved, reset patience counter
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter > patience:
            break

        # Tell Optuna that we can prune here if needed
        trial.report(val_accuracy, epoch)
        # If the trial should be pruned, you can raise an exception
        if trial.should_prune():
            raise optuna.TrialPruned()

    # We want to maximize accuracy
    return best_val_accuracy


######################################################
# Running Optuna study
######################################################
if __name__ == "__main__":
    # Optional: specify a storage, sampler, pruner, etc.
    study = optuna.create_study(
        direction="maximize",  # since we're returning accuracy
        storage=OPTUNA_SAVE_LOG,
        study_name=OPTUNA_STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    # Try a certain number of trials
    n_trials = OPTUNA_N_TRIALS
    study.optimize(objective, n_trials=n_trials, timeout=None)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (best accuracy): {best_trial.value}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # If you want to re-train a final model with the best hyperparams:
    best_params = best_trial.params
    print("\nRe-training final model with best parameters:")
    pprint(best_params)

    # Example: final training with best hyperparams
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    train_batch_size = best_params["train_batch_size"]
    val_batch_size = best_params["val_batch_size"]

    # Build final model
    final_model = ShallowFBCSPNet(
        n_channels,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto").to(device)

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(train_batch_size, val_batch_size)

    # Define optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = 30  # you may use more epochs if you want
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    patience = 6
    counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, final_model, loss_fn, optimizer, scheduler, epoch, device
        )
        val_loss, val_accuracy = test_model(val_loader, final_model, loss_fn, phase="val")
        test_loss, test_accuracy = test_model(test_loader, final_model, loss_fn, phase="test")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(final_model, optimizer, scheduler, epoch, extra_file_str="best_")
            print("Test ACC : ", test_accuracy , "At Epoch : ", epoch)
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter > patience:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in validation accuracy for {patience} consecutive epochs.")
            save_model(final_model, optimizer, scheduler, epoch, extra_file_str="earlyStop_")
            break

    print("Optuna hyperparameter search + final training complete.")
