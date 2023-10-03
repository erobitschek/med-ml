import torch as torch
import torch.nn as nn


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
