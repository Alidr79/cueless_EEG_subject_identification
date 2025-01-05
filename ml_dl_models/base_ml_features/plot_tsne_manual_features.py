import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mne_features.feature_extraction import FeatureExtractor




# Load datasets
all_dataset = np.load('../all_dataset.npy', allow_pickle=True).item()
eeg_all = all_dataset['eeg_data']

all_metadata = pd.read_csv('../all_metadata.csv')

test_indices = all_metadata[all_metadata['session_id'] == 5].index.values
eeg_test = eeg_all[test_indices]


test_metadata = all_metadata[all_metadata['session_id'] == 5].reset_index(drop=True)


# For Wavelet-based features
# fe = FeatureExtractor(sfreq=250,
#                                          selected_funcs=[
#                                                          'wavelet_coef_energy'])


# For statistical-based features

fe = FeatureExtractor(sfreq=250,
                                         selected_funcs=[
                                                         'mean', 'variance', 'skewness', 'kurtosis'])

features = fe.fit_transform(eeg_test)
print("shape features :", features.shape)




tsne = TSNE(n_components=2, random_state=42)
tsne_space = tsne.fit_transform(features)

unique_subjects = test_metadata['subject_id'].unique()
cmap = cm.get_cmap('tab20', len(unique_subjects))

plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(15, 15))
for i, sub in enumerate(unique_subjects):
    color = cmap(i)

    indices = test_metadata[test_metadata['subject_id'] == sub].index.values
    plt.scatter(tsne_space[indices, 0], tsne_space[indices, 1], label=f"sub-{sub}", c=[color], alpha=1.)


plt.legend(loc = "upper left")
plt.xlabel("tSNE_1")
plt.ylabel("tSNE_2")
plt.title("tSNE plot of statistical-based feature set")
plt.savefig('tsne_stats(colors_subjects).png')