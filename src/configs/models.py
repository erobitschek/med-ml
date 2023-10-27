from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy.typing as npt
import torch as torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression as skLogisticRegression


class TorchLogisticRegression(nn.Module):
    """A simple logistic regression implementation using PyTorch.

    Attributes:
        linear: A linear layer that transforms the input data.
        input_dim: Dimensionality (i.e., number of features) of the input data.

    Example:
        >>> model = LogisticRegression(input_dim=10)
        >>> sample_input = torch.rand((5, 10))
        >>> output = model(sample_input)

    Args:
        input_dim: Dimensionality (i.e., number of features) of the input data.
    """

    def __init__(self, input_dim):
        super(TorchLogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.linear = nn.Linear(input_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the logistic regression model.

        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            Output tensor after passing through the linear layer and the sigmoid activation. Shape: (batch_size, 1).
        """
        out = torch.sigmoid(self.linear(x))
        return out

    def attributes(self):
        """Returns the attributes of the model as a dictionary."""
        return {"input_dim": self.input_dim, "output_dim": self.output_dim}


class SimpleCNN(nn.Module):
    """Implements a simple CNN model using PyTorch.

    Attributes:
        input_dim: Shape of the input tensor (batch, input_dims, sequence length).
        n_class: Number of classes.
        filter_sizes: Filter depth for each Conv1d layer.
        kernel_sizes: Kernel size for each Conv1d layer.
        pooling_sizes: Pooling window size for each layer.
        dense_sizes: Size of each output sample for linear layers.
        conv1d_layers: Convolutional layers of the network.
        pooling_layers: Pooling layers of the network.
        adaptive_max_pool: Adaptive max pooling layer to ensure dimensional compatibility.
        linear_layers: Linear layers of the network.
        output_layer: Final layer of the network that outputs class probabilities.
        relu: ReLU activation function.
    """

    def __init__(
        self,
        input_dim: npt.ArrayLike,
        n_class: int,
        filter_sizes: List[int] = [16, 32, 32, 64],
        kernel_sizes: List[int] = [5, 3, 3, 3],
        pooling_sizes: List[int] = [2, 2, 2, 2],
        dense_sizes: List[int] = [64, 64],
    ):
        """
        Parameters
        ----------
        input_dim: shape of input (batch, input_dims, sequence length)
        n_class: number of classes to be categorized
        filter_sizes: filter depth for Conv1d layers
        kernel_sizes: kernel sizes for Conv1d layers
        pooling_sizes: pooling window sizes
        dense_sizes: size of each output sample for linear layers
        """

        super(SimpleCNN, self).__init__()

        self.input_dim = input_dim
        self.n_class = n_class

        # Define network architecture
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.dense_sizes = dense_sizes

        # Define convolutional layers
        conv1d_layers = []
        for in_ch, out_ch, kernel_size in zip(
            [
                self.input_dim[1],
            ]
            + self.filter_sizes[:-1],
            self.filter_sizes,
            self.kernel_sizes,
        ):
            conv1d_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=0))
        self.conv1d_layers = nn.ModuleList(conv1d_layers)

        # Define pooling layers
        pooling_layers = []
        for pool_size in self.pooling_sizes:
            pooling_layers.append(nn.MaxPool1d(pool_size))
        self.pooling_layers = pooling_layers

        # Add an adaptive max pool for dimensional compatibility (for smooth transition between conv1d and linear layers)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # Define linear layers (assumes that the first linear layer follows last conv1d layer)
        linear_layers = []
        for in_feat, out_feat in zip(
            [
                self.filter_sizes[-1],
            ]
            + self.dense_sizes[:-1],
            self.dense_sizes,
        ):
            linear_layers.append(nn.Linear(in_feat, out_feat))

        self.linear_layers = nn.ModuleList(linear_layers)

        # Define output layer
        self.output_layer = nn.Linear(self.dense_sizes[-1], self.n_class)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """Defines the forward pass for the SimpleCNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Network output tensor.
        """
        # Process through convolutional and pooling layers.
        for c_layer, c_pool in zip(self.conv1d_layers, self.pooling_layers):
            x = c_layer(x)
            x = self.relu(x)
            x = c_pool(x)

        # Transition from convolutional to linear layers.
        x = self.adaptive_max_pool(x)
        x = torch.squeeze(x, dim=-1)

        # Process through linear layers.
        for c_layer in self.linear_layers:
            x = c_layer(x)
            x = self.relu(x)

        # Process through output layer and apply softmax
        x = self.output_layer(x)

        # Return network response
        return x

    def predict_proba(self, x: torch.Tensor):
        """Instead of raw, unnormalized scores, returns the predicted probabilities of each class"""
        with torch.no_grad():
            outp = self.forward(x)
        return nn.functional.softmax(outp, dim=1)

    def attributes(self):
        """Returns the attributes of the model as a dictionary."""
        return {
            "input_dim": self.input_dim,
            "n_class": self.n_class,
            "filter_sizes": self.filter_sizes,
            "kernel_sizes": self.kernel_sizes,
            "pooling_sizes": self.pooling_sizes,
            "dense_sizes": self.dense_sizes,
        }


class SimpleRNN(nn.Module):
    """Implements a basic RNN model using PyTorch.

    Attributes:
        input_dim: Shape of the input tensor.
        num_layers: Number of RNN layers.
        hidden_size: Size of the RNN hidden state.
        sequence_length: Length of the sequence.
        n_class: Number of classes.
        rnn: The core RNN module.
        flatten: Flatten layer.
        fc: Fully connected layer for output.
    """

    def __init__(
        self,
        input_dim: npt.ArrayLike,
        n_class: int,
        sequence_length: int,
        num_layers: int = 3,
        hidden_size: int = 64,
    ):
        super(SimpleRNN, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_class = n_class
        self.rnn = torch.nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.flatten = torch.nn.Flatten()
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, self.n_class)

    def forward(self, input):
        """Defines the forward pass for the SimpleRNN.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Network output tensor.
        """
        # Re-order dimensions to: batch - sequence (time) - features
        x = input.permute(0, 2, 1)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward prop through time
        out, _ = self.rnn(x, h0)

        # Only take last hidden state to pass through the output layer (could consider others in future)
        x = self.fc(self.flatten(out))
        return x

    def predict_proba(self, x):
        """Instead of raw, unnormalized scores, returns the predicted probabilities of each class"""
        with torch.no_grad():
            outp = self.forward(x)
        return nn.functional.softmax(outp, dim=1)

    def attributes(self):
        """Returns the attributes of the model as a dictionary."""
        return {
            "input_dim": self.input_dim,
            "num_layers": self.num_layers,
            "n_class": self.n_class,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
        }


class ModelType(Enum):
    LOGREG_SKLEARN = skLogisticRegression
    LOGREG_TORCH = TorchLogisticRegression
    LGBM_CLASSIFIER = LGBMClassifier
    SIMPLE_CNN = SimpleCNN
    SIMPLE_RNN = SimpleRNN
    # Add other models here as needed

    def __call__(self, *args, **kwargs) -> Callable:
        # This allows the enum instance itself to be called as a constructor, forwarding arguments to the actual constructor
        return self.value(*args, **kwargs)

    @staticmethod
    def from_name(model_name: str):
        for model in ModelType:
            if model.name == model_name.upper():
                return model
        raise ValueError(f"Unsupported model: {model_name}")
