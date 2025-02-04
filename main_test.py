import os
import json
import argparse

from DataLoader import ECGDataLoader
from train_test_eval.test_and_eval import test_and_eval
from utils.plot_utils import plot_FP_FN_signals


config_path = 'cfg/test_cfg.json'

def main():
    parser = argparse.ArgumentParser(description="Test an ECG model using configuration file.")
    parser.add_argument("config_path", default=config_path, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as file:
        train_config = json.load(file)

    # Extract config parameters
    model_path = train_config.get("model_path")
    model_name = train_config.get("model_name")
    X_array_path = train_config.get("X_array_path")
    y_df_path = train_config.get("y_df_path")
    output_folder = train_config.get("output_folder", "output_test")
    os.makedirs(output_folder, exist_ok=True)

    # Load Data
    loader_config = train_config.get("data_loader_cfg", {})
    loader = ECGDataLoader(loader_config)

    if X_array_path is not None and y_df_path is not None:
        X, y_labels = loader.read_arrays(X_array_path, y_df_path)
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
    preds = test_and_eval(X, y_labels, model_path, model_name=model_name, test=True, output_folder=output_folder)
    plot_FP_FN_signals(X, y_labels, model_path, preds, test=True)


if __name__ == "__main__":
    main()
