import mne
import pandas as pd
from tqdm import tqdm
import numpy as np


# No extra log for MNE
mne.set_log_level('WARNING') 

# Define paths and parameters

all_subject_ids = range(1, 12)
session_ids = [1,2,3,4,5]
data_path_template = "../Dataset/derivatives/preprocessed_eeg/sub-{:02d}/ses-{:02d}/eeg/sub-{:02d}_ses-{:02d}_eeg.fif"

# Load and preprocess each subject's se = []
all_epochs = []
# len_each_train_subject = []
for subject_id in tqdm(all_subject_ids):
    subject_epochs = []
    len_subject = 0
    for session_id in session_ids:
        raw_path = data_path_template.format(subject_id, session_id, subject_id, session_id)
        # Load raw data
        raw = mne.io.read_raw_fif(raw_path, preload=True)

        # Define events and epochs
        events, event_id = mne.events_from_annotations(raw)
        tmin, tmax = 0, 1.995  # define your epoch time range

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline = None, preload=True)
#         epochs.info['subject_info'] = {'subject_id':subject_id, 'session_id':session_id} # not working for session_id
        metadata = pd.DataFrame({'subject_id': [subject_id] * len(epochs), 'session_id': [session_id] * len(epochs)})
        epochs.metadata = metadata
        len_subject += len(epochs)
        subject_epochs.append(epochs)


    # Concatenate epochs for the current subject
#     len_each_train_subject.append(len_subject)
    subject_epochs = mne.concatenate_epochs(subject_epochs)
    all_epochs.append(subject_epochs)




all_metadata = pd.concat([all_epochs[i].metadata for i in range(len(all_epochs))], axis = 0).reset_index(drop = True)
words = np.concatenate([all_epochs[i].events[:,-1] for i in range(len(all_epochs))], axis = -1)
all_metadata['words'] = words


all_eeg = np.concatenate([all_epochs[i].get_data('eeg') for i in range(len(all_epochs))])

print("shape all_eeg", all_eeg.shape)
print("shape all_metadata", all_metadata.shape)


all_dict = {
    'eeg_data': all_eeg,
    'subject': all_metadata.loc[:, 'subject_id'].values -1
}
np.save('all_dataset.npy', all_dict)

all_metadata.to_csv('all_metadata.csv', index = False)
np.save('all_eeg.npy', all_eeg)
