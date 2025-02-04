import os
import wfdb
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample, find_peaks
from sklearn.model_selection import train_test_split


class ECGDataLoader:
    def __init__(self, target_fs=300, target_length=3000, signal_lead='I', min_max_norm=False, pass_filter='hpf'):
        """
        Initializes the ECGDataLoader with specified parameters for preprocessing ECG signals.

        Parameters:
        - target_fs (int, optional): Target sampling frequency of the ECG signal in Hz. Default is 300 Hz.
        - target_length (int, optional): Target length of ECG signals (in number of samples). Signals will be trimmed or padded to match this length. Default is 3000 samples.
        - signal_lead (str, optional): The lead from which the ECG signal is extracted. Options include:
          'I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'. Default is 'I'.
        - min_max_norm (bool, optional): Whether to apply min-max normalization to the ECG signals. Default is False.
        - pass_filter (str, optional): Type of filtering applied to the signal. Options:
          'hpf' (high-pass filter), 'bpf' (band-pass filter), or None. Default is 'hpf'.

        Attributes:
        - X (list): Stores the processed ECG signals.
        - y_labels (dict): Dictionary to store corresponding labels for ECG signals.
        - min_size (int or None): Stores the minimum size of the ECG signals in the dataset (before processing).
        - max_size (int or None): Stores the maximum size of the ECG signals in the dataset (before processing).
        - min_max_norm (bool): Stores whether min-max normalization is enabled.
        - filter_method (str): Stores the selected filtering method.
        """
        self.signal_lead = signal_lead
        self.target_fs = target_fs
        self.fixed_length = target_length
        self.X = []
        self.y_labels = {}
        self.min_size = None
        self.max_size = None
        self.min_max_norm = min_max_norm
        self.filter_method = pass_filter


    def load_data(self, folder_path, labels_path, data_type, origin_freq=300, normalize=True, resample_freq=False, augment=False):
        """
        Loads and preprocesses ECG signals from the specified folder.

        Parameters:
        - folder_path (str): Directory containing ECG signal files.
        - labels_path (str): Path to the labels file.
        - data_type (str): Format of the data files ('dat' or 'mat').
        - origin_freq (int): Original sampling frequency (default: 300 Hz).
        - normalize (bool): Apply normalization (default: True).
        - resample_freq (bool/int): Resample frequency if specified.
        - augment (bool): Perform data augmentation on positive samples.

        Process:
        - Reads labels and ECG signals.
        - Preprocesses signals (resampling, normalization).
        - Augments positive samples if enabled.
        - Stores processed signals and updates metadata.
        """
        all_ids = self._load_labels(labels_path)

        for sig_i, sig_id in enumerate(tqdm(all_ids, "Loading data")):
            label = self.y_labels[sig_id]['label']
            if data_type == 'dat':
                signal = self._read_dat_files(folder_path, sig_id)
            elif data_type == 'mat':
                signal = self._read_mat_files(folder_path, sig_id)

            signal, no_var_flag = self._preprocess_signal(signal, origin_freq=origin_freq, resample_freq=resample_freq,
                                                          normalize=normalize)

            if no_var_flag:
                self.y_labels[sig_id]['label'] = 1

            if augment == True and label == 1 and len(signal) > 2 * self.fixed_length:
                # augment only on positives samples with long enough signal
                num_segments = len(signal) // self.fixed_length
                for i in range(num_segments):
                    start = i * int(len(signal)/num_segments)
                    end = start + int(len(signal)/num_segments)
                    if end <= len(signal):
                        segment = signal[start:end]
                    elif end > len(signal) and len(signal[start:]) > self.fixed_length:
                        segment = signal[start:]
                    else:
                        continue

                    if self.fixed_length:
                        segment = self._trim_or_pad(segment)

                    self.X.append(segment)
                    self.y_labels[sig_id]['aug'] = True
                    if 'X_index' in list(self.y_labels[sig_id].keys()):
                        self.y_labels[sig_id]['X_index'].append(len(self.X) - 1)
                    else:
                        self.y_labels[sig_id]['X_index'] = [len(self.X) - 1]

            else:
                # Trim or pad if needed
                if self.fixed_length:
                    signal = self._trim_or_pad(signal)
                self.X.append(signal)
                self.y_labels[sig_id]['aug'] = False
                self.y_labels[sig_id]['X_index'] = [len(self.X) - 1]

    def _load_labels(self, labels_path):
        """
        Loads signal labels from specified files.

        Parameters:
        - labels_path (list): A list containing either:
          - Two text files (acceptable & unacceptable records).
          - A single CSV file mapping signals to labels.

        Process:
        - Reads signal IDs and assigns labels (0 for clean, 1 for noisy).
        - Supports both text-based and CSV-based labeling formats.

        Returns:
        - list: All signal IDs found in the label files.
        """
        all_ids = []
        if len(labels_path) == 2:
            # Read labels from RECORDS-acceptable & RECORDS-unacceptable
            acceptable_ids = self._read_record_file(labels_path[0])
            unacceptable_ids = self._read_record_file(labels_path[1])
            all_ids = acceptable_ids + unacceptable_ids
            for sig_id in all_ids:
                if sig_id in acceptable_ids:
                    self.y_labels[sig_id] = {'label': 0}
                else:
                    self.y_labels[sig_id] = {'label': 1}
        elif len(labels_path) == 1:
            # Read labels from CSV
            labels_df = pd.read_csv(labels_path[0], header=None)
            for _, row in labels_df.iterrows():
                all_ids.append(row[0])
                if "~" in row[1]:
                    self.y_labels[row[0]] = {'label': 1}
                else:
                    self.y_labels[row[0]] = {'label': 0}
        else:
            raise ValueError(
                "labels_path must be a list of either single csv path or 2 text files path describing the non-noisy and noisy data accordingly")

        return all_ids

    def _read_dat_files(self, folder_path, sig_id):
        """ Read single-lead ECG from a CINC 2011 .dat file. """
        record_path = os.path.join(folder_path, sig_id)
        signal, fields = wfdb.rdsamp(record_path)
        lead_index = fields['sig_name'].index(self.signal_lead)

        return signal[:, lead_index]

    def _read_mat_files(self, folder_path, sig_id):
        """ Read single-lead ECG from a CINC 2017 .mat file. """
        mat_path = os.path.join(folder_path, f"{sig_id}.mat")
        mat_data = scipy.io.loadmat(mat_path)

        return mat_data["val"][0]

    def _preprocess_signal(self, signal, origin_freq, resample_freq=False, normalize=True):
        """ Resample, normalize, and trim/pad the ECG signal. """
        no_var_flag = False

        # HPF (<4 Hz)
        def highpass_filter(sig, fs, cutoff=4, order=4):
            nyquist = 0.5 * fs
            low = cutoff / nyquist
            b, a = butter(order, low, btype='high')
            return filtfilt(b, a, sig)

        # BPF (4-40 Hz)
        def bandpass_filter(sig, fs, lowcut=4, highcut=40, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, sig)

        # Resample if needed
        if resample_freq:
            original_length = len(signal)
            target_length = int(original_length * (self.target_fs / origin_freq))
            signal = resample(signal, target_length)
            origin_freq = self.target_fs

        # Apply bandpass filter
        if self.filter_method == 'bpf':
            signal = bandpass_filter(signal, origin_freq)
        elif self.filter_method == 'hpf':
            signal = highpass_filter(signal, origin_freq)

        # Update min/max sizes
        self.min_size = min(self.min_size or len(signal), len(signal))
        self.max_size = max(self.max_size or len(signal), len(signal))

        if normalize:
            if self.min_max_norm:
                # Min-Max normalization to range [-1, 1]
                min_val, max_val = np.min(signal), np.max(signal)
                if max_val != min_val:
                    signal = 2 * (signal - min_val) / (max_val - min_val) - 1
                else:
                    no_var_flag = True
            else:
                # Standard normalization (zero mean, unit variance)
                if np.std(signal) != 0:
                    signal = (signal - np.mean(signal)) / np.std(signal)
                else:
                    no_var_flag = True

        return signal, no_var_flag

    def _read_record_file(self, file_path):
        """ Read RECORDS-acceptable or RECORDS-unacceptable files. """
        if not os.path.exists(file_path):
            raise ValueError(f"File path - {file_path} does not exists")
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _trim_or_pad(self, signal):
        """Trim or pad the signal to the target length by taking the middle part when trimming.

        Args:
            signal (np.ndarray): The input 1D signal.
        Returns:
            np.ndarray: Trimmed or padded signal.
        """
        current_length = len(signal)
        target_length = self.fixed_length

        if current_length > target_length:
            start = (current_length - target_length) // 2
            end = start + target_length
            return signal[start:end]

        elif current_length < target_length:
            # Pad symmetrically
            pad_left = (target_length - current_length) // 2
            pad_right = target_length - (current_length + pad_left)
            return np.pad(signal, (pad_left, pad_right), mode='constant')

        return signal

    def finalize_data(self):
        """ Transform the data into to np.array and the tag\labeled data to a pandas df
        (columns- id (str), label (binary-number), augmented (binary), x_index(int))"""
        X = np.vstack(self.X)
        y_labels = pd.DataFrame([
                {'id': key, 'label': value['label'], 'augmented': value['aug'], 'x_index': x_idx}
                for key, value in self.y_labels.items()
                for x_idx in value['X_index']  # Expand X_index list
            ])
        return X, y_labels

    def train_test_split(self, df, sub_sample_CINC_2017=False, noise_ratio=0.2, train_ratio=0.8, cv_k=5):
        """
        Splits the dataset into train and test sets with stratification.

        Parameters:
        - df (DataFrame): Input data containing signal IDs and labels.
        - sub_sample_CINC_2017 (bool): Whether to downsample not-noisy CINC_2017.
        - noise_ratio (float): Target noise-valid ratio in the dataset.
        - train_ratio (float): Proportion of data to use for training.
        - cv_k (int): Number of cross-validation folds.

        Process:
        - Optionally removes negative (not noisy\valid) excess CINC_2017 samples to match the noise ratio.
        - Performs stratified train-test split across multiple folds.

        Returns:
        - DataFrame: Updated dataset with fold assignments.
        """
        unique_ids = df[['id', 'label']].drop_duplicates()

        # Create fold columns
        for fold in range(cv_k):
            unique_ids[f'fold_{fold}'] = np.nan  # Default all values to NaN

            if sub_sample_CINC_2017:
                # Select only A0 samples (CINC 2017) with label=0
                a0_samples = unique_ids[(unique_ids['label'] == 0) & (unique_ids['id'].str.startswith('A0'))]

                # Calculate how many A0 samples need to be ignored
                num_noisy = len(unique_ids[unique_ids['label'] == 1])
                target_non_noisy = int(num_noisy / noise_ratio * (1 - noise_ratio))
                num_to_ignore = len(a0_samples) - target_non_noisy

                # Randomly select the IDs to ignore
                drop_ids = np.random.choice(a0_samples['id'], num_to_ignore, replace=False)
            else:
                drop_ids = []

            # Remaining data after ignoring drop_ids
            valid_ids = unique_ids[~unique_ids['id'].isin(drop_ids)]

            # Stratified train-test split (ensuring augmented samples remain together)
            train_ids, test_ids = train_test_split(valid_ids, train_size=train_ratio, stratify=valid_ids['label'],
                                                   random_state=fold)
            # Assign the fold column values
            unique_ids.loc[unique_ids['id'].isin(train_ids['id']), f'fold_{fold}'] = 'train'
            unique_ids.loc[unique_ids['id'].isin(test_ids['id']), f'fold_{fold}'] = 'test'

        # Merge back to the original y_labels dataframe
        df = df.merge(unique_ids, on=['id', 'label'], how='left')

        return df

    def plot_ecg_segment(self, seg_id):
        """ Plot a ECG segment given a sample ID"""
        plt.figure()
        x_ids = self.y_labels[seg_id]['X_index']
        label = self.y_labels[seg_id]['label']

        plt.plot(self.X[x_ids[0]])
        plt.title(f'Signal {seg_id}, Noisy: {label}')
        plt.xticks([])
        plt.show()

    def analyze(self, y_label_df, plot_samples=0, output_folder=None):
        """
        Analyzes signal quality by computing peaks and signal-to-noise ratio (SNR).

        Parameters:
        - y_label_df (DataFrame): Data containing signal IDs and labels.
        - plot_samples (int): Number of random samples to visualize (default: 0).
        - output_folder (str): Folder to save histograms (optional).

        Process:
        - Computes SNR and peak count for each signal.
        - Optionally visualizes a subset of signals.
        - Generates histograms of computed features.

        Returns:
        - DataFrame: Updated with computed SNR and peak counts.
        """
        y_label_df['peaks'] = 0
        y_label_df['snr'] = 0
        if plot_samples > 0:
            pos_samples = y_label_df[y_label_df.label == 1]['id'].sample(plot_samples).to_list()
            neg_samples = y_label_df[y_label_df.label == 0]['id'].sample(plot_samples).to_list()
        else:
            pos_samples, neg_samples = [], []
        plot_samples = pos_samples + neg_samples
        for row_i, row in y_label_df.iterrows():
            x = self.X[row.x_index]
            snr = self._compute_snr(x)
            peaks, _ = find_peaks(x)
            y_label_df.at[row_i, 'snr'] = snr
            y_label_df.at[row_i, 'peaks'] = len(peaks)
            if row['id'] in plot_samples:
                plt.figure()
                plt.plot(x)
                plt.plot(peaks, x[peaks], "x")
                plt.plot(np.zeros_like(x), "--", color="gray")
                plt.title(f"ID:{row['id']}, Label: {'Noisy' if row['label'] == 1 else 'Valid'}, "
                          f"Peaks:{len(peaks)}, SNR:{snr}")
                plt.show()

        self._plot_histograms(y_label_df, output_folder)
        return y_label_df

    def _plot_histograms(self, y_label_df, output_folder=None):
        """
        Plots histograms of peak counts for different labels.

        Parameters:
        - y_label_df (DataFrame): Data containing peak counts and labels.
        - output_folder (str): Folder to save the histogram plots (optional).

        Process:
        - Generates histograms for each label separately.
        - Creates an overlayed histogram for comparison.
        - Saves the plot if an output folder is provided.
        """
        label_0 = y_label_df[y_label_df["label"] == 0]["peaks"]
        label_1 = y_label_df[y_label_df["label"] == 1]["peaks"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.histplot(label_0, bins=30, kde=True, ax=axes[0], color="blue")
        axes[0].set_title("Histogram of Peaks (Valid)")

        sns.histplot(label_1, bins=30, kde=True, ax=axes[1], color="red")
        axes[1].set_title("Histogram of Peaks (Noisy)")

        sns.histplot(label_0, bins=30, kde=True, stat='density', ax=axes[2], color="blue", label="Valid", alpha=0.6)
        sns.histplot(label_1, bins=30, kde=True, stat='density', ax=axes[2], color="red", label="Noisy", alpha=0.6)
        axes[2].set_title("Overlayed Histogram of Peaks")
        axes[2].legend()

        plt.tight_layout()
        self._save_and_close_plot(fig, "histograms.png", output_folder)

    def _save_and_close_plot(self, fig, filename, output_folder=None, close_time=5):
        """ Saves the plot and closes it after a pause. """
        filepath = f"{output_folder}/{filename}"
        if output_folder:
            plt.savefig(filepath)
        plt.pause(close_time)
        plt.close(fig)

    def save_arrays(self, X_array, y_label_df, output_path):
        """ Saves feature array and labels to output_path. """
        np.save(os.path.join(output_path,'X_array.npy'), X_array)
        y_label_df.to_csv(os.path.join(output_path, 'y_label_df.csv'), index=False)

    def read_arrays(self, X_array_path, y_label_df_path):
        """ Loads feature array and labels from disk. """
        y_label_df = pd.read_csv(y_label_df_path)
        X_array = np.load(X_array_path)
        return X_array, y_label_df

    @staticmethod
    def _compute_snr(signal):
        """ Computes the signal-to-noise ratio (SNR). """
        signal_power = np.mean(signal ** 2)
        noise_power = np.var(signal - np.mean(signal))
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
