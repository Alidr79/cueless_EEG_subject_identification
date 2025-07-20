import numpy as np
import pandas as pd 

eeg_data = np.load("all_eeg.npy")
metadata = pd.read_csv("all_metadata.csv")


#### 
## Generate train_dataset for session-1
###

train_indices = metadata[metadata['session_id'] == 1].index.values
eeg_train = eeg_data[train_indices]
labels_train = metadata.loc[train_indices, 'subject_id'].values

train_dict = {
    'eeg_data': eeg_train,
    'subject': labels_train -1
}
np.save('dataset_ses-1.npy', train_dict)


#### 
## Generate train_dataset for session-2
###

train_indices = metadata[metadata['session_id'] == 2].index.values
eeg_train = eeg_data[train_indices]
labels_train = metadata.loc[train_indices, 'subject_id'].values

train_dict = {
    'eeg_data': eeg_train,
    'subject': labels_train -1
}
np.save('dataset_ses-2.npy', train_dict)



#### 
## Generate train_dataset for session-3
###

train_indices = metadata[metadata['session_id'] == 3].index.values
eeg_train = eeg_data[train_indices]
labels_train = metadata.loc[train_indices, 'subject_id'].values

train_dict = {
    'eeg_data': eeg_train,
    'subject': labels_train -1
}
np.save('dataset_ses-3.npy', train_dict)