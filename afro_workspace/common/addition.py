import numpy as np
import os
import matplotlib.pyplot as plt
import torch
def plot_history(train_history, num_epochs, ckpt_dir, seed, validation_history=None):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key] for summary in train_history]
        plt.plot(np.linspace(0, num_epochs, len(train_history)), train_values, label='train')
        if validation_history is not None:
            val_values = [summary[key] for summary in validation_history]
            plt.plot(np.linspace(0, num_epochs, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)

def calculate_average_metrics(history_list):
    if not history_list:
        return {}

    all_keys = history_list[0].keys()

    summed_metrics = {key: 0.0 for key in all_keys}
    
    num_records = len(history_list)

    for record in history_list:
        for key, value in record.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            summed_metrics[key] += value

    average_metrics = {}
    for key, total_sum in summed_metrics.items():
        average_metrics[key] = total_sum / num_records

    return average_metrics