import os

import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, out_dir=None):
    """
    Plot training and validation losses over epochs and optionally save the plot to a specified directory.

    Parameters
    ----------
    train_losses : list of float
        List of training losses, typically one value per epoch.
    val_losses : list of float
        List of validation losses, typically one value per epoch.
    out_dir : str, optional
        Path to the directory where the plot should be saved.
        If not provided, the plot is displayed but not saved.

    Returns
    -------
    None

    Example
    -------
    >>> train_losses = [0.5, 0.4, 0.3, 0.2]
    >>> val_losses = [0.6, 0.5, 0.35, 0.25]
    >>> plot_loss(train_losses, val_losses, out_dir='./plots')
    """

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(train_losses, label="Training Loss", color="blue", linewidth=1.5)
    ax.plot(val_losses, label="Validation Loss", color="red", linewidth=1.5)

    ax.set_title("Training and Validation Losses Over Epochs", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/loss_plot.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
