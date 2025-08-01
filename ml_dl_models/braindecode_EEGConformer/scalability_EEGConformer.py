import random
import os
import numpy as np
import pandas as pd
import torch
from braindecode.models import EEGConformer
from braindecode.util import set_random_seeds
import optuna
from optuna.trial import Trial
from pprint import pprint
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import time
from datetime import datetime
import json


# Configuration
BASE_DATA_PATH = '../'
RESULTS_DIR = 'scalability_results'
OPTUNA_N_TRIALS = 30
N_EPOCHS = 30
PATIENCE = 6

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


def reset_all_seeds():
    control_randomness(42)
    set_random_seeds(seed=seed, cuda=cuda)
    g = torch.Generator()
    g.manual_seed(42)

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


def create_subset_datasets(n_subjects, base_path=BASE_DATA_PATH):    
    # Load original data
    eeg_data = np.load(os.path.join(base_path, "all_eeg.npy"))
    metadata = pd.read_csv(os.path.join(base_path, "all_metadata.csv"))
    
    # Filter data to include only first n_subjects (subject_id 1 to n_subjects)
    subject_mask = metadata['subject_id'] <= n_subjects
    filtered_metadata = metadata[subject_mask].copy()
    filtered_indices = filtered_metadata.index.values
    filtered_eeg_data = eeg_data[filtered_indices]
    
    # Create train dataset (sessions 1,2,3)
    train_indices = filtered_metadata[filtered_metadata['session_id'].isin([1,2,3])].index.values
    eeg_train = eeg_data[train_indices]
    labels_train = filtered_metadata.loc[train_indices, 'subject_id'].values
    
    train_dict = {
        'eeg_data': eeg_train,
        'subject': labels_train - 1  # Convert to 0-indexed
    }
    
    # Create val dataset (session 4)
    val_indices = filtered_metadata[filtered_metadata['session_id'] == 4].index.values
    eeg_val = eeg_data[val_indices]
    labels_val = filtered_metadata.loc[val_indices, 'subject_id'].values
    
    val_dict = {
        'eeg_data': eeg_val,
        'subject': labels_val - 1  # Convert to 0-indexed
    }
    
    # Create test dataset (session 5)
    test_indices = filtered_metadata[filtered_metadata['session_id'] == 5].index.values
    eeg_test = eeg_data[test_indices]
    labels_test = filtered_metadata.loc[test_indices, 'subject_id'].values
    
    test_dict = {
        'eeg_data': eeg_test,
        'subject': labels_test - 1  # Convert to 0-indexed
    }
    
    # Save datasets
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_path = os.path.join(RESULTS_DIR, f'train_dataset_{n_subjects}subjects.npy')
    val_path = os.path.join(RESULTS_DIR, f'val_dataset_{n_subjects}subjects.npy')
    test_path = os.path.join(RESULTS_DIR, f'test_dataset_{n_subjects}subjects.npy')
    
    np.save(train_path, train_dict)
    np.save(val_path, val_dict)
    np.save(test_path, test_dict)
    
    return train_path, val_path, test_path, len(eeg_train), len(eeg_val), len(eeg_test)


def get_dataloaders(train_path, val_path, test_path, train_batch_size, val_batch_size, test_batch_size=16):
    """Create dataloaders from dataset paths."""
    train_dataset = NpyDataset(train_path)
    val_dataset = NpyDataset(val_path)
    test_dataset = NpyDataset(test_path)
    
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
# Training and Evaluation Functions
######################################################
def train_one_epoch(
        dataloader: DataLoader, model: Module, loss_fn, optimizer,
        scheduler: LRScheduler, epoch: int, device, print_batch_stats=True
):
    model.train()
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

    scheduler.step()
    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model(
    dataloader: DataLoader, model: Module, loss_fn, print_batch_stats=True, phase="val"
):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader)) if print_batch_stats else enumerate(dataloader)

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
                f"{phase} Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {batch_loss:.6f}"
            )

    average_loss = total_loss / n_batches
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    if print_batch_stats:
        print(f"{phase} Accuracy: {100 * accuracy:.2f}%")
        print(f"{phase} Macro Precision: {100 * macro_precision:.2f}%")
        print(f"{phase} Macro Recall: {100 * macro_recall:.2f}%")
        print(f"{phase} Loss: {average_loss:.6f}\n")

    return average_loss, accuracy, macro_precision, macro_recall


######################################################
# Optuna Objective Function
######################################################
def objective(trial: Trial, n_classes, train_path, val_path, test_path):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-2)
    train_batch_size = trial.suggest_categorical("train_batch_size", [16, 32, 64, 128])
    val_batch_size = trial.suggest_categorical("val_batch_size", [16, 32])

    # Create model
    model = EEGConformer(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        sfreq=250,
        final_fc_length=1080
        ).to(device)

    # Create Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, train_batch_size, val_batch_size)

    # Define optimizer, scheduler, loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS - 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device, print_batch_stats=False
        )
        val_loss, val_accuracy, val_precision, val_recall = test_model(val_loader, model, loss_fn, print_batch_stats=False, phase="val")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter > PATIENCE:
            break

        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_accuracy


def train_final_model(n_classes, best_params, train_path, val_path, test_path):    
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    train_batch_size = best_params["train_batch_size"]
    val_batch_size = best_params["val_batch_size"]

    # Build final model
    final_model = EEGConformer(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        sfreq=250,
        final_fc_length=1080
            ).to(device)

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, train_batch_size, val_batch_size)

    # Define optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS - 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_test_accuracy = 0.0
    best_test_precision = 0.0
    best_test_recall = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_epoch = 0
    counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, final_model, loss_fn, optimizer, scheduler, epoch, device, print_batch_stats=False
        )
        val_loss, val_accuracy, val_precision, val_recall = test_model(val_loader, final_model, loss_fn, print_batch_stats=False, phase="val")
        test_loss, test_accuracy, test_precision, test_recall = test_model(test_loader, final_model, loss_fn, print_batch_stats=False, phase="test")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_test_accuracy = test_accuracy
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_epoch = epoch
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter > PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return (best_val_accuracy, best_val_precision, best_val_recall, 
            best_test_accuracy, best_test_precision, best_test_recall, best_epoch)


def run_scalability_experiment():
    """Run the complete scalability experiment."""
    
    results = []
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Starting Scalability Analysis: Performance vs Number of Subjects")
    print("=" * 70)
    
    for n_subjects in range(2, 11):  # 2 to 10 subjects
        
        reset_all_seeds()
        
        print(f"\n{'='*50}")
        print(f"EXPERIMENT: {n_subjects} SUBJECTS")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Create datasets for current number of subjects
        print(f"Creating datasets for {n_subjects} subjects...")
        train_path, val_path, test_path, n_train, n_val, n_test = create_subset_datasets(n_subjects)
        
        print(f"Dataset sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        # Setup Optuna study
        study_name = f"study_EEGConformer_{n_subjects}subjects"
        storage_path = f"sqlite:///{RESULTS_DIR}/{study_name}.sqlite3"
        
        study = optuna.create_study(
            direction="maximize",
            storage=storage_path,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Run hyperparameter optimization
        print(f"Starting hyperparameter optimization with {OPTUNA_N_TRIALS} trials...")
        study.optimize(
            lambda trial: objective(trial, n_subjects, train_path, val_path, test_path), 
            n_trials=OPTUNA_N_TRIALS, 
            timeout=None
        )
        
        best_params = study.best_trial.params
        best_val_acc_optuna = study.best_trial.value
        
        print(f"Best validation accuracy from Optuna: {best_val_acc_optuna:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        print("Training final model with best parameters...")
        reset_all_seeds()
        
        (final_val_acc, final_val_precision, final_val_recall, 
         final_test_acc, final_test_precision, final_test_recall, best_epoch) = train_final_model(
            n_subjects, best_params, train_path, val_path, test_path
        )
        
        # Calculate experiment duration
        duration = time.time() - start_time
        
        # Store results
        result = {
            'n_subjects': n_subjects,
            'n_train_samples': n_train,
            'n_val_samples': n_val,
            'n_test_samples': n_test,
            'best_val_accuracy': final_val_acc,
            'best_val_precision': final_val_precision,
            'best_val_recall': final_val_recall,
            'best_test_accuracy': final_test_acc,
            'best_test_precision': final_test_precision,
            'best_test_recall': final_test_recall,
            'best_epoch': best_epoch,
            'optuna_trials': len(study.trials),
            'best_hyperparams': best_params,
            'duration_minutes': duration / 60,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        print(f"Final Results for {n_subjects} subjects:")
        print(f"  - Validation Accuracy: {final_val_acc:.4f}")
        print(f"  - Validation Precision: {final_val_precision:.4f}")
        print(f"  - Validation Recall: {final_val_recall:.4f}")
        print(f"  - Test Accuracy: {final_test_acc:.4f}")
        print(f"  - Test Precision: {final_test_precision:.4f}")
        print(f"  - Test Recall: {final_test_recall:.4f}")
        print(f"  - Best Epoch: {best_epoch}")
        print(f"  - Duration: {duration/60:.2f} minutes")
        
        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, 'scalability_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def generate_markdown_report(results):
    """Generate a comprehensive markdown report."""
    
    report_path = os.path.join(RESULTS_DIR, 'scalability_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Scalability Analysis Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write("This report analyzes how the EEGConformer model performance scales ")
        f.write("with an increasing number of subjects (classes) from 2 to 11.\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Model:** EEGConformer\n")
        f.write(f"- **Input Channels:** {n_channels}\n")
        f.write(f"- **Input Window Samples:** {input_window_samples}\n")
        f.write(f"- **Optuna Trials per Subject:** {OPTUNA_N_TRIALS}\n")
        f.write(f"- **Max Epochs:** {N_EPOCHS}\n")
        f.write(f"- **Early Stopping Patience:** {PATIENCE}\n")
        f.write(f"- **Device:** {device}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Subjects | Train Samples | Val Samples | Test Samples | Val Acc | Val Prec | Val Rec | Test Acc | Test Prec | Test Rec | Best Epoch | Duration (min) |\n")
        f.write("|----------|---------------|-------------|--------------|---------|----------|---------|----------|-----------|----------|------------|----------------|\n")
        
        for result in results:
            f.write(f"| {result['n_subjects']} | {result['n_train_samples']} | {result['n_val_samples']} | ")
            f.write(f"{result['n_test_samples']} | {result['best_val_accuracy']:.4f} | {result['best_val_precision']:.4f} | ")
            f.write(f"{result['best_val_recall']:.4f} | {result['best_test_accuracy']:.4f} | {result['best_test_precision']:.4f} | ")
            f.write(f"{result['best_test_recall']:.4f} | {result['best_epoch']} | {result['duration_minutes']:.1f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for result in results:
            f.write(f"### {result['n_subjects']} Subjects\n\n")
            f.write(f"- **Dataset Sizes:**\n")
            f.write(f"  - Training: {result['n_train_samples']} samples\n")
            f.write(f"  - Validation: {result['n_val_samples']} samples\n")
            f.write(f"  - Testing: {result['n_test_samples']} samples\n")
            f.write(f"- **Performance:**\n")
            f.write(f"  - Validation Accuracy: {result['best_val_accuracy']:.4f} ({result['best_val_accuracy']*100:.2f}%)\n")
            f.write(f"  - Validation Precision: {result['best_val_precision']:.4f} ({result['best_val_precision']*100:.2f}%)\n")
            f.write(f"  - Validation Recall: {result['best_val_recall']:.4f} ({result['best_val_recall']*100:.2f}%)\n")
            f.write(f"  - Test Accuracy: {result['best_test_accuracy']:.4f} ({result['best_test_accuracy']*100:.2f}%)\n")
            f.write(f"  - Test Precision: {result['best_test_precision']:.4f} ({result['best_test_precision']*100:.2f}%)\n")
            f.write(f"  - Test Recall: {result['best_test_recall']:.4f} ({result['best_test_recall']*100:.2f}%)\n")
            f.write(f"  - Best Epoch: {result['best_epoch']}\n")
            f.write(f"- **Training Details:**\n")
            f.write(f"  - Optuna Trials Completed: {result['optuna_trials']}\n")
            f.write(f"  - Training Duration: {result['duration_minutes']:.2f} minutes\n")
            f.write(f"- **Best Hyperparameters:**\n")
            for param, value in result['best_hyperparams'].items():
                if isinstance(value, float):
                    f.write(f"  - {param}: {value:.2e}\n")
                else:
                    f.write(f"  - {param}: {value}\n")
            f.write(f"- **Timestamp:** {result['timestamp']}\n\n")
        
        f.write("## Analysis\n\n")
        f.write("### Performance Trends\n\n")
        
        # Calculate some basic statistics
        test_accuracies = [r['best_test_accuracy'] for r in results]
        test_precisions = [r['best_test_precision'] for r in results]
        test_recalls = [r['best_test_recall'] for r in results]
        val_accuracies = [r['best_val_accuracy'] for r in results]
        val_precisions = [r['best_val_precision'] for r in results]
        val_recalls = [r['best_val_recall'] for r in results]
        
        f.write(f"- **Test Performance Range:**\n")
        f.write(f"  - Accuracy: {min(test_accuracies):.4f} - {max(test_accuracies):.4f}\n")
        f.write(f"  - Precision: {min(test_precisions):.4f} - {max(test_precisions):.4f}\n")
        f.write(f"  - Recall: {min(test_recalls):.4f} - {max(test_recalls):.4f}\n")
        f.write(f"- **Validation Performance Range:**\n")
        f.write(f"  - Accuracy: {min(val_accuracies):.4f} - {max(val_accuracies):.4f}\n")
        f.write(f"  - Precision: {min(val_precisions):.4f} - {max(val_precisions):.4f}\n")
        f.write(f"  - Recall: {min(val_recalls):.4f} - {max(val_recalls):.4f}\n")
        f.write(f"- **Average Performance:**\n")
        f.write(f"  - Test Accuracy: {np.mean(test_accuracies):.4f}\n")
        f.write(f"  - Test Precision: {np.mean(test_precisions):.4f}\n")
        f.write(f"  - Test Recall: {np.mean(test_recalls):.4f}\n")
        f.write(f"  - Validation Accuracy: {np.mean(val_accuracies):.4f}\n")
        f.write(f"  - Validation Precision: {np.mean(val_precisions):.4f}\n")
        f.write(f"  - Validation Recall: {np.mean(val_recalls):.4f}\n\n")
        
        # Find best and worst performing configurations
        best_idx = np.argmax(test_accuracies)
        worst_idx = np.argmin(test_accuracies)
        
        f.write(f"### Best Performance\n")
        f.write(f"- **{results[best_idx]['n_subjects']} subjects** achieved the highest test accuracy: ")
        f.write(f"{results[best_idx]['best_test_accuracy']:.4f} ({results[best_idx]['best_test_accuracy']*100:.2f}%)\n")
        f.write(f"  - Test Precision: {results[best_idx]['best_test_precision']:.4f} ({results[best_idx]['best_test_precision']*100:.2f}%)\n")
        f.write(f"  - Test Recall: {results[best_idx]['best_test_recall']:.4f} ({results[best_idx]['best_test_recall']*100:.2f}%)\n\n")
        
        f.write(f"### Lowest Performance\n")
        f.write(f"- **{results[worst_idx]['n_subjects']} subjects** achieved the lowest test accuracy: ")
        f.write(f"{results[worst_idx]['best_test_accuracy']:.4f} ({results[worst_idx]['best_test_accuracy']*100:.2f}%)\n")
        f.write(f"  - Test Precision: {results[worst_idx]['best_test_precision']:.4f} ({results[worst_idx]['best_test_precision']*100:.2f}%)\n")
        f.write(f"  - Test Recall: {results[worst_idx]['best_test_recall']:.4f} ({results[worst_idx]['best_test_recall']*100:.2f}%)\n\n")
        
        f.write("### Notes\n\n")
        f.write("1. All experiments used the same random seed for reproducibility\n")
        f.write("2. Early stopping was applied based on validation accuracy\n")
        f.write("3. Hyperparameters were optimized using Optuna with TPE sampler\n")
        f.write("4. Results show performance on the test set using the model with best validation accuracy\n")
    
    print(f"\nMarkdown report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    print("Starting Scalability Analysis...")
    print(f"Results will be saved in: {RESULTS_DIR}")
    
    # Run the complete experiment
    results = run_scalability_experiment()
    
    # Generate markdown report
    report_path = generate_markdown_report(results)
    
    print("\n" + "="*70)
    print("SCALABILITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Markdown report: {report_path}")
    print(f"JSON results: {os.path.join(RESULTS_DIR, 'scalability_results.json')}")
    
    # Print summary
    print("\nQuick Summary:")
    for result in results:
        print(f"  {result['n_subjects']} subjects: Test Acc = {result['best_test_accuracy']:.4f} ({result['best_test_accuracy']*100:.2f}%), "
              f"Prec = {result['best_test_precision']:.4f} ({result['best_test_precision']*100:.2f}%), "
              f"Rec = {result['best_test_recall']:.4f} ({result['best_test_recall']*100:.2f}%)")