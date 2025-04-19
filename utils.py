import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

sns.set_style("ticks")

def read_models(models_json_path):
    """
    Reads a JSON file containing model information and returns a dictionary.
    
    Args:
        models_json_path (str): Path to the JSON file containing model information.

    Returns:
        dict: Dictionary containing model information.
    """
    with open(models_json_path, 'r') as json_file:
        models = json.load(json_file)
    
    for key in models.keys():
        models[key] = np.array(models[key])
    
    return models


def plot_loss(loss, val_loss, window=4, palette=sns.color_palette("flare")):
    
    loss_avg = np.array([np.mean(loss[i:i+window]) for i in range(len(loss) - window + 1)])
    val_loss_avg = np.array([np.mean(val_loss[i:i+window]) for i in range(len(val_loss) - window + 1)])
    loss_std = np.array([np.std(loss[i:i+window]) for i in range(len(loss) - window + 1)])
    val_loss_std = np.array([np.std(val_loss[i:i+window]) for i in range(len(val_loss) - window + 1)])

    epochs = np.arange(len(loss_avg)) + (window - 1) / 2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(epochs, loss_avg, label='Training Loss', color=palette[0], marker='.', markersize=5)
    ax.fill_between(epochs, loss_avg - loss_std, loss_avg + loss_std, color=palette[0], alpha=0.2)
    ax.plot(epochs, val_loss_avg, label='Validation Loss', color=palette[-1], marker='.', markersize=5)
    ax.fill_between(epochs, val_loss_avg - val_loss_std, val_loss_avg + val_loss_std, color=palette[-1], alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend(frameon=False)
    
    sns.despine()

    plt.show()