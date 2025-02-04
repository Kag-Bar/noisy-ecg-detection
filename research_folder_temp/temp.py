import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataLoader import ECGDataLoader

# Usage
loader = ECGDataLoader()
# loader.load_data(
#     folder_path='set-a',
#     labels_path=['set-a\RECORDS-acceptable', 'set-a\RECORDS-unacceptable'],
#     data_type='dat',
#     origin_freq=500,
#     normalize=True,
#     resample_freq=True,
#     augment=False)
#
# loader.load_data(
#     folder_path='training2017',
#     labels_path=['training2017/REFERENCE-original.csv'],
#     data_type='mat',
#     origin_freq=300,
#     normalize=True,
#     resample_freq=False,
#     augment=False)
#
# y_labels = pd.DataFrame([
#     {'id': key, 'label': value['label'], 'augmented': value['aug'], 'x_index': x_idx}
#     for key, value in loader.y_labels.items()
#     for x_idx in value['X_index']  # Expand X_index list
# ])

def plot_random_samples(df, X, y_dict = None):
    if y_dict:
        selected = y_dict
        fig, axes = plt.subplots(5, 2, figsize=(10, 15))
        axes = axes.flatten()

        for i, key in enumerate(y_dict.keys()):
            x_index = y_dict[key]
            x_vector = X[x_index]
            if not isinstance(x_vector, list):
                x_vector = x_vector.tolist()
            sample_id = key
            label = 1 if i <= 4 else 0
            length = len(x_vector)
            subplot_pos = 2*i if i<=4 else 2*(i-5) +1
            axes[subplot_pos].plot(x_vector)
            axes[subplot_pos].set_title(
                f"ID: {sample_id}, Label: {'Noisy' if label == 1 else 'Valid'} (Length: {length})")
            axes[subplot_pos].set_xticks([])
            axes[subplot_pos].set_yticks([])
    else:
        # Separate indices for each label
        pos_indices = df[df['label'] == 1].index.to_list()
        neg_indices = df[df['label'] == 0].index.to_list()

        # Randomly select 5 from each
        selected_pos = np.random.choice(pos_indices, 5, replace=False)
        selected_neg = np.random.choice(neg_indices, 5, replace=False)

        # Create subplots
        fig, axes = plt.subplots(5, 2, figsize=(10, 15))
        axes = axes.flatten()
        selected = {}

        for i, idx in enumerate(selected_pos):
            x_index = df.loc[idx, 'x_index']
            x_vector = X[x_index]
            if not isinstance(x_vector, list):
                x_vector = list(x_vector)
            sample_id = df.loc[idx, 'id']
            selected[sample_id] = x_index
            label = df.loc[idx, 'label']
            length = len(x_vector)

            axes[i*2].plot(x_vector)
            axes[i*2].set_title(f"ID: {sample_id}, Label: {'Noisy' if label == 1 else 'Valid'} (Length: {length})")
            axes[i*2].set_xticks([])
            axes[i*2].set_yticks([])

        for i, idx in enumerate(selected_neg):
            x_index = df.loc[idx, 'x_index']
            x_vector = X[x_index]
            if not isinstance(x_vector, list):
                x_vector = list(x_vector)
            sample_id = df.loc[idx, 'id']
            selected[sample_id] = x_index
            label = df.loc[idx, 'label']
            length = len(x_vector)

            axes[i*2 +1].plot(x_vector)
            axes[i*2 +1].set_title(f"ID: {sample_id}, Label: {'Noisy' if label == 1 else 'Valid'} (Length: {length})")
            axes[i*2 + 1].set_xticks([])
            axes[i*2 + 1].set_yticks([])

    plt.tight_layout()
    plt.show()

    return selected

###
y_dict = {'1156063': 800,
 'A06435': 7432,
 '1086219': 790,
 '2117407': 909,
 'A04452': 5449,
 'A00201': 1198,
 'A08007': 9004,
 'A02619': 3616,
 'A06194': 7191,
 'A01222': 2219}
###


# selected = plot_random_samples(y_labels, loader.X, y_dict)

# Access processed data
# X, y_labels = loader.finalize_data()
# y_labels = loader.train_test_split(y_labels,True, 0.4)
# y_labels = loader.analyze(y_labels)
# loader.save_arrays(X, y_labels, 'output/HPF')


X, df = loader.read_arrays('output/HPF/X_array_0.4_noise_ratio_hpf.npy', 'output/HPF/y_label_df_0.4_noise_ratio_hpf.csv')
loader.X = X
loader.analyze(df)

import seaborn as sns

def signal_to_image(signal):
    return signal.reshape(50, 60)


def plot_signal_heatmaps(y_dict, df, X):
    num_signals = len(y_dict)
    cols = min(5, num_signals)  # Maximum 5 columns
    rows = (num_signals + cols - 1) // cols  # Calculate rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_signals > 1 else [axes]

    for i, key in enumerate(y_dict.keys()):
        x_index = df[df.id == key]['x_index'].values[0]
        signal = X[x_index]
        img = signal_to_image(signal)

        sns.heatmap(img, cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=False, ax=axes[i])
        axes[i].set_title(f"ID: {key}, Label: {'Noisy' if i<=4 else 'Valid'}")

    # Hide unused subplots
    for i in range(num_signals, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# Call the function
plot_signal_heatmaps(y_dict, df, X)
