import pandas as pd
from tqdm import tqdm
import numpy as np

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

model = MOMENTPipeline.from_pretrained(
                                        "AutonLab/MOMENT-1-base", 
                                        model_kwargs={
                                            'task_name': 'classification',
                                            'n_channels': 30,
                                            'num_class': 11,
                                            'freeze_encoder': True,
                                            'freeze_embedder': True,
                                            'freeze_head': True,
                                            'reduction': 'mean',
                                        },
                                        local_files_only=False
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
            output = model(batch_x, reduction = "mean")
            #mean over patches dimension, [batch_size x d_model]
            embedding = output.embeddings.mean(dim=1)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(output.logits.argmax(dim=1).cpu().numpy())        

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels


# Create Dataset
all_dataset = PaddedNpyDataset('../all_dataset.npy')

# Create DataLoader
all_loader = DataLoader(all_dataset, batch_size = 1, shuffle = False)

device = 'cuda'
checkpoint = torch.load('checkpoints_base_val-4_test-5/model_epoch_best_32.pth')
model.load_state_dict(checkpoint['model_state_dict'])
all_embeddings, all_labels = get_embeddings(model, device, all_loader)

np.save('32-shot_base_embeddings.npy', all_embeddings)
np.save('32-shot_base_labels.npy', all_labels)