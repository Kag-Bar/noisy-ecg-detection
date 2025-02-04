import os.path

from DataLoader import ECGDataLoader
from train_and_eval import train_and_evaluate
from train_and_eval_CNN import train_and_evaluate_CNN

train_method = 'CNN'
X_array_path = 'output/X_array.npy'
y_df_path = 'output/y_label_df.csv'
output_folder = 'output/train'

# Load Data
loader = ECGDataLoader()
if X_array_path is not None and y_df_path is not None:
    X, y_labels = loader.read_arrays('output/X_array.npy', 'output/y_label_df.csv')
else:
    loader.load_data(
        folder_path='set-a',
        labels_path=['set-a\RECORDS-acceptable', 'set-a\RECORDS-unacceptable'],
        data_type='dat',
        origin_freq=500,
        normalize=True,
        resample_freq=True,
        augment=False)

    loader.load_data(
        folder_path='training2017',
        labels_path=['training2017/REFERENCE-original.csv'],
        data_type='mat',
        origin_freq=300,
        normalize=True,
        resample_freq=False,
        augment=True)

    # Access processed data
    X, y_labels = loader.finalize_data()
    y_labels = loader.train_test_split(y_labels, sub_sample_CINC_2017=True, noise_ratio=0.4)
    y_labels = loader.analyze(y_labels, plot_samples=3, output_folder=output_folder)
    loader.save_arrays(X, y_labels, output_folder)

if train_method == 'CNN':
    # Train and evaluate CNN pipeline
    result_df = train_and_evaluate_CNN(X, y_labels, output_folder, batch_size=128, epochs=20)
    result_df.to_csv(os.path.join(output_folder, 'result_df.csv'), index=False)
else:
    # Train and Evaluate with MLP
    train_and_evaluate(X, y_labels, output_folder)
