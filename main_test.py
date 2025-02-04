from DataLoader import ECGDataLoader
from test_and_eval import test_and_eval
from plot_utils import plot_FP_FN_signals


# Initialize
train_method = 'resnet18'
X_array_path = 'output_test/X_array_test.npy'
y_df_path = 'output_test/y_label_df_test.csv'
output_folder = 'output_test'
model_path = 'output_test/best_model_fold_3.pth'

# Load Data
loader = ECGDataLoader()
if X_array_path is not None and y_df_path is not None:
    X, y_labels = loader.read_arrays('output/X_array.npy', 'output/y_label_df.csv')
else:
    loader = ECGDataLoader()
    loader.load_data(
        folder_path='sample2017/validation',
        labels_path=['sample2017/validation/REFERENCE.csv'],
        data_type='mat',
        origin_freq=300,
        normalize=True,
        resample_freq=False,
        augment=False)


    # Analyze and save data arrays
    X, y_labels = loader.finalize_data()
    y_labels = loader.analyze(y_labels, output_folder=output_folder)
    loader.save_arrays(X, y_labels, output_folder)

# Predict with model
preds = test_and_eval(X, y_labels, model_path, model_name=model_path, test=True, output_folder=output_folder)
plot_FP_FN_signals(X, y_labels, model_path, preds, test=True)