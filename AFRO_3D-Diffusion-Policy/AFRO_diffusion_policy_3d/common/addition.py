import numpy as np
import os
import matplotlib.pyplot as plt
import torch
def plot_history(train_history, num_epochs, ckpt_dir, seed, validation_history=None):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key] for summary in train_history]
        plt.plot(np.linspace(0, num_epochs, len(train_history)), train_values, label='train')
        if validation_history is not None:
            val_values = [summary[key] for summary in validation_history]
            plt.plot(np.linspace(0, num_epochs, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)

def calculate_average_metrics(history_list):
    """
    将历史记录列表中所有字典的相同键的值进行平均，并返回一个包含平均值的新列表。

    Args:
        history_list (list): 包含指标字典的列表。
                             例如: [{'loss': 0.85, 'mse': 0.50}, {'loss': 0.42, 'mse': 0.25}]

    Returns:
        list: 包含一个字典的列表，该字典包含所有指标的平均值。
    """
    if not history_list:
        return {}

    # 1. 识别所有指标的键
    # 假设所有字典的键都相同
    all_keys = history_list[0].keys()
    
    # 用于累加每个指标的总和
    summed_metrics = {key: 0.0 for key in all_keys}
    
    num_records = len(history_list)

    # 2. 累加所有记录中的指标值
    for record in history_list:
        for key, value in record.items():
            # 确保值是数值类型，如果是 PyTorch Tensor，需要先转换
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            # 累加总和
            summed_metrics[key] += value

    # 3. 计算平均值
    average_metrics = {}
    for key, total_sum in summed_metrics.items():
        average_metrics[key] = total_sum / num_records

    # 4. 按照您的要求，返回一个包含这个平均字典的列表
    return average_metrics