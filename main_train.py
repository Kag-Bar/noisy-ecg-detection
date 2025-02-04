import os
import json
import argparse

from DataLoader import ECGDataLoader
from train_test_eval.train_and_eval import train_and_evaluate
from train_test_eval.train_and_eval_CNN import train_and_evaluate_CNN

config_path = 'cfg/train_cfg.json'

def main():
    parser = argparse.ArgumentParser(description="Train ECG model using configuration file.")
    parser.add_argument("config_path", default=config_path, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as file:
        train_config = json.load(file)

    # Extract config parameters
    train_method = train_config.get("train_method", "CNN")
    X_array_path = train_config.get("X_array_path")
    y_df_path = train_config.get("y_df_path")
    output_folder = train_config.get("output_folder", "output")
    os.makedirs(output_folder, exist_ok=True)

    # Load Data
    loader_config = train_config.get("data_loader_cfg", {})
    loader = ECGDataLoader(loader_config)

    if X_array_path and y_df_path:
        X, y_labels = loader.read_arrays(X_array_path, y_df_path)
    else:
        loader.load_data(
            folder_path="set-a",
            labels_path=["set-a/RECORDS-acceptable", "set-a/RECORDS-unacceptable"],
            data_type="dat",
            origin_freq=500,
            normalize=True,
            resample_freq=True,
            augment=False
        )

        loader.load_data(
            folder_path="training2017",
            labels_path=["training2017/REFERENCE-original.csv"],
            data_type="mat",
            origin_freq=300,
            normalize=True,
            resample_freq=False,
            augment=True
        )

        X, y_labels = loader.finalize_data()
        y_labels = loader.train_test_split(
            y_labels,
            sub_sample_CINC_2017=loader_config.get("sub_sample_CINC_2017", True),
            noise_ratio=loader_config.get("noise_ratio", 0.4)
        )
        y_labels = loader.analyze(y_labels, plot_samples=loader_config.get("plot_samples", 3),
                                  output_folder=output_folder)
        loader.save_arrays(X, y_labels, output_folder)

    # Train model
    model_config = train_config.get("training_cfg", {})
    if train_method == "CNN":
        result_df = train_and_evaluate_CNN(
            X, y_labels, output_folder,
            batch_size=model_config.get("batch_size", 128),
            epochs=model_config.get("epochs", 20)
        )
        result_df.to_csv(os.path.join(output_folder, "result_df.csv"), index=False)
    else:
        train_and_evaluate(X, y_labels, output_folder)


if __name__ == "__main__":
    main()
