from momentfm import MOMENTPipeline
import random
import os 
import torch 
import numpy as np 

def control_randomness(seed: int = 42):
    """Function to control randomness in the code."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

control_randomness(42)


# For dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)


model = MOMENTPipeline.from_pretrained(
                                        "AutonLab/MOMENT-1-small", 
                                        model_kwargs={
                                            'task_name': 'classification',
                                            'n_channels': 30,
                                            'num_class': 11,
                                            'freeze_encoder': False,
                                            'freeze_embedder': False,
                                            'reduction': 'mean',
                                        },
                                        )
model.init()
model.to("cuda")
print(model)
# summary(model, input_size=(16, 30, 512))
# def print_model_with_frozen_status(model):
#     for name, module in model.named_modules():
#         is_frozen = not any(param.requires_grad for param in module.parameters())
#         print(f"Layer: {name}, Frozen: {is_frozen}")
# print("\nLayer Frozen Status:")
# print_model_with_frozen_status(model)


from pprint import pprint
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader



class PaddedNpyDataset(Dataset):
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
        
        # Pad the third dimension (500) to 512 using numpy.pad
        padded_sample = np.pad(sample, ((0, 0), (0, 12)), mode='constant')
        
        # Convert the padded sample to a torch tensor
        padded_sample = torch.tensor(padded_sample, dtype=torch.float)
        
        # Convert the label to a tensor (optional: depending on its type)
        label = torch.tensor(label, dtype=torch.long)  # Assuming labels are integer class indices
        
        # Return both the padded sample and the label
        return padded_sample, label


def get_embeddings(model, device, dataloader: DataLoader):
    '''
    labels: [num_samples]
    embeddings: [num_samples x d_model]
    '''
    embeddings, labels = [], []
    model.eval()

    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
            # [batch_size x 12 x 512]
            batch_x = batch_x.to(device).float()
            # [batch_size x num_patches x d_model (=1024)]
            output = model(batch_x)
            #mean over patches dimension, [batch_size x d_model]
            embedding = output.embeddings.mean(dim=1)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)        

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels


# Create Dataset
train_dataset = PaddedNpyDataset('../train_dataset.npy')
val_dataset = PaddedNpyDataset('../val_dataset.npy')
test_dataset = PaddedNpyDataset('../test_dataset.npy')

print("train data len:", len(train_dataset))
print("val data len:", len(val_dataset))
print("test data len:", len(test_dataset))


# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 8, worker_init_fn=seed_worker,generator=g)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False, num_workers = 2, worker_init_fn=seed_worker,generator=g)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False, num_workers = 2, worker_init_fn=seed_worker,generator=g)



def train_epoch(model, device, train_dataloader, criterion, optimizer, scheduler, reduction='mean'):
    model.to(device)
    model.train()
    losses = []
    total_correct = 0

    for batch_x, batch_labels in tqdm(train_dataloader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device).float()
        batch_labels = batch_labels.to(device)

        #note that since MOMENT encoder is based on T5, it might experiences numerical unstable issue with float16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
            output = model(batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
            total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()

        loss.backward()

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    accuracy = total_correct / len(train_dataloader.dataset)
    return avg_loss, accuracy



def evaluate_epoch(dataloader, model, criterion, device, phase='val', reduction='mean'):
    model.eval()
    model.to(device)
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader):
            batch_x = batch_x.to(device).float()
            batch_labels = batch_labels.to(device)

            output = model(batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
            total_loss += loss.item()
            total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy



def save_model(model, optimizer, scheduler, epoch, save_path="checkpoints_small_val-4_test-5", extra_file_str = ""):
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

# Function to load the model
# def load_model(model, optimizer, scheduler, load_path):
#     checkpoint = torch.load(load_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     epoch = checkpoint['epoch']
#     print(f"Model loaded from {load_path} at epoch {epoch}")
#     return epoch

def log_write(file, log_input):
    with open(file, 'a') as log_file:
        log_file.write(log_input)


# # load checkpoint continue learning
# checkpoint = torch.load('checkpoints/model_epoch_100.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# del checkpoint

# start_epoch = 100

log_to_file = 'finetune_100_epochs_small_val-4_test-5_log.md'
epoch = 100

log_write(log_to_file, f"train on ses-1-2-3 val on ses-4 and test on ses-5\n")

log_write(log_to_file, f"train data len: {len(train_dataset)}\n")
log_write(log_to_file, f"val data len: {len(val_dataset)}\n")
log_write(log_to_file, f"test data len: {len(test_dataset)}\n")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=epoch * len(train_loader))
device = 'cuda'

val_loss, val_accuracy = evaluate_epoch(val_loader, model, criterion, device, phase='val')
print(f'Epoch {0} val loss: {val_loss}, val accuracy: {val_accuracy}')
log_write(log_to_file, f'Epoch {0} val loss: {val_loss}, val accuracy: {val_accuracy}\n')

test_loss, test_accuracy = evaluate_epoch(test_loader, model, criterion, device, phase='test')
print(f'Epoch {0} test loss: {test_loss}, test accuracy: {test_accuracy}')
log_write(log_to_file, f'Epoch {0} test loss: {test_loss}, test accuracy: {test_accuracy}\n')

best_acc = val_accuracy
patience = 15
counter = 0 

for i in tqdm(range(1, epoch +1)):
    
    train_loss, train_accuracy = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)
    val_loss, val_accuracy = evaluate_epoch(val_loader, model, criterion, device, phase='val')
    test_loss, test_accuracy = evaluate_epoch(test_loader, model, criterion, device, phase='test')

    print(f'Epoch {i}, train loss: {train_loss}, train accuracy: {train_accuracy}, val loss: {val_loss}, val accuracy: {val_accuracy}, test loss:{test_loss}, test accuracy:{test_accuracy}, lr: {scheduler.get_last_lr()[0]}')
    log_write(log_to_file, f'Epoch {i}, train loss: {train_loss}, train accuracy: {train_accuracy}, val loss: {val_loss}, val accuracy: {val_accuracy}, test loss:{test_loss}, test accuracy:{test_accuracy}, lr: {scheduler.get_last_lr()[0]}\n')
    


    if val_accuracy > best_acc:
        best_acc = val_accuracy
        save_model(model, optimizer, scheduler, epoch = i, extra_file_str = "best_")
        counter = 0

    else:
        counter += 1  # Increment counter if no improvement
    
    # Early stopping condition
    if counter > patience:
        print(f"Early stopping triggered at epoch {i}. No improvement in validation accuracy for {patience} consecutive epochs.")
        save_model(model, optimizer, scheduler, epoch = i, extra_file_str = "earlyStop_")
        break

    if (i % 20 == 0):
        print("save model")
        save_model(model, optimizer, scheduler, epoch = i)