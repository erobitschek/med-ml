import os

import matplotlib.pyplot as plt


def plot_loss(train_losses: list[float], val_losses: list[float], out_dir=None) -> None:
    """Plots training and validation losses over epochs. If an output directory is provided, the plot is saved, 
    otherwise it is displayed.

    Args:
        train_losses: Training losses, typically one value per epoch.
        val_losses: Validation losses, typically one value per epoch.
        out_dir: Directory where the plot should be saved. If not specified, 
            the plot will be displayed but not saved.
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
