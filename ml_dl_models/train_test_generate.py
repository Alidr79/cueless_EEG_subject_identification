import numpy as np
import pandas as pd 

eeg_data = np.load("all_eeg.npy")
metadata = pd.read_csv("all_metadata.csv")

train_indices = metadata[metadata['session_id'].isin([1,2,3])].index.values
eeg_train = eeg_data[train_indices]
labels_train = metadata.loc[train_indices, 'subject_id'].values

train_dict = {
    'eeg_data': eeg_train,
    'subject': labels_train -1
}
np.save('train_dataset.npy', train_dict)




#### 
## Generate train_dataset for session-1,2
###

train_indices = metadata[metadata['session_id'].isin([1,2])].index.values
eeg_train = eeg_data[train_indices]
labels_train = metadata.loc[train_indices, 'subject_id'].values

train_dict = {
    'eeg_data': eeg_train,
    'subject': labels_train -1
}
np.save('train_dataset_ses-1,2.npy', train_dict)



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
np.save('train_dataset_ses-1.npy', train_dict)





val_indices = metadata[metadata['session_id'] == 4].index.values
eeg_val = eeg_data[val_indices]
labels_val = metadata.loc[val_indices, 'subject_id'].values

val_dict = {
    'eeg_data': eeg_val,
    'subject': labels_val -1
}
np.save('val_dataset.npy', val_dict)


test_indices = metadata[metadata['session_id'] == 5].index.values
eeg_test = eeg_data[test_indices]
labels_test = metadata.loc[test_indices, 'subject_id'].values

test_dict = {
    'eeg_data': eeg_test,
    'subject': labels_test -1
}
np.save('test_dataset.npy', test_dict)