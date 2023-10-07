import itertools
import os

from typing import Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


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


def plot_confusion_matrix(
    run_dir: str,
    true_labels: npt.ArrayLike,
    predictions: npt.ArrayLike,
    classes: list = ["0", "1"],
    normalize: Optional[bool] = False,
    title: str="confusion_matrix",
    cmap: plt.cm = plt.cm.YlGnBu,
):
    """
    Helper function to plot a confusion matrix with the ability to normalize the output with `normalize=True`.

    Parameters
    ----------
    run_dir: path to the directory containing the predictions.txt file and where the plot should be saved
    classes: a list of the possible classes for the test set
    normalize: produces a normalized version of the heatmap (sometimes more informative)
    title: name of the plot to put in the figure
    cmap: sets the colormap to use for the heatmap
    """
    cm = confusion_matrix(true_labels, predictions)
    fmt = "d"

    if normalize:
        fmt = ".2f"
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "confusion_matrix_normalized"

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{run_dir}{title}.pdf", format="pdf", bbox_inches="tight", dpi=300)
