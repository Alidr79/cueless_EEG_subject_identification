import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


embeddings = np.load('zero-shot_large_embeddings.npy')
print("embedding shape:", embeddings.shape)

all_metadata = pd.read_csv('../all_metadata.csv')

test_indices = all_metadata[all_metadata['session_id'] == 5].index.values
test_embeddings = embeddings[test_indices]

test_metadata = all_metadata[all_metadata['session_id'] == 5].reset_index(drop=True)


tsne = TSNE(n_components=2, random_state=42)
tsne_space = tsne.fit_transform(test_embeddings)

print("tsne shape: ", tsne_space.shape)

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
plt.title("tSNE plot of zero-shot MOMENT-large test embeddings")
plt.savefig('zero-shot_large_embedding(colors_subjects).png')