import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics_individual(log_dir, csv_file_name='metrics.csv'):
    """
    Reads the CSV file containing training metrics and generates individual plots for each metric.

    :param log_dir: Path to the directory containing the metrics CSV file
    :param csv_file_name: Name of the CSV file with metrics (default: 'metrics.csv')
    """
    # construct the full path to the CSV file
    csv_file_path = os.path.join(log_dir, csv_file_name)

    # check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Metrics file not found at {csv_file_path}")
        return

    # read the CSV file into a pandas DataFrame
    metrics_df = pd.read_csv(csv_file_path)

    epochs = metrics_df['epoch']

    # iterate over all columns except 'epoch' to plot each metric
    for metric in metrics_df.columns:
        if metric == 'epoch':
            continue

        # extract metric values
        metric_values = metrics_df[metric]

        # create a new figure for each metric
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, metric_values, label=metric, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(f'{metric} over Epochs')
        plt.legend(loc='best')
        plt.grid(True)

        # dynamically adjust y-axis limits to fit the data range
        if metric_values.min() != metric_values.max():  # Avoid identical limits for flat lines
            plt.ylim(metric_values.min() * 0.9, metric_values.max() * 1.1)

        # save the plot in the log directory
        plot_save_path = os.path.join(log_dir, f'{metric}_plot.png')
        plt.savefig(plot_save_path)
        plt.close()

