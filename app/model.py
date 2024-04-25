from typing import Tuple

import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float)


class FamusNetTwo(nn.Module):
    def __init__(self, input_size):
        super(FamusNetTwo, self).__init__()
        self.act = nn.SiLU()

        self.layers = nn.Sequential()
        self.layers.add_module("fc0", nn.Linear(input_size, 4096))
        self.layers.add_module("act1", self.act)
        self.layers.add_module("fc1", nn.Linear(4096, 1024))
        self.layers.add_module("act2", self.act)
        self.layers.add_module("fc2", nn.Linear(1024, 512))
        self.layers.add_module("act3", self.act)
        self.layers.add_module("fc3", nn.Linear(512, 320))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single tensor.
        Can be used after training to get the embedding of a tensor.
        :param x: A tensor.
        :return: The embedding of the tensor.
        """
        x = self.layers(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(
        self, triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of a triplet.
        :param triplet: A tuple of (ancor, positive, negative) tensors.
        :return: A tuple of (ancor, positive, negative) embeddings.
        """
        ancor, positive, negative = triplet[0], triplet[1], triplet[2]
        output1 = self.forward_once(ancor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)
        return output1, output2, output3


class FamusNet(nn.Module):
    def __init__(self, input_size):
        super(FamusNet, self).__init__()
        self.act = nn.SiLU()

        self.layers = nn.Sequential()
        self.layers.add_module("fc0", nn.Linear(input_size, 4096))
        self.layers.add_module("act1", self.act)
        self.layers.add_module("fc1", nn.Linear(4096, 1000))
        self.layers.add_module("act2", self.act)
        self.layers.add_module("fc2", nn.Linear(1000, 1000))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single tensor.
        Can be used after training to get the embedding of a tensor.
        :param x: A tensor.
        :return: The embedding of the tensor.
        """
        x = self.layers(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(
        self, triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of a triplet.
        :param triplet: A tuple of (ancor, positive, negative) tensors.
        :return: A tuple of (ancor, positive, negative) embeddings.
        """
        ancor, positive, negative = triplet[0], triplet[1], triplet[2]
        output1 = self.forward_once(ancor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)
        return output1, output2, output3


class VariableNet(nn.Module):
    """A variable number of layers, with a fixed embedding size.
    The input size is the size of the input vector, and the embedding size is the size of the output vector.
    The number of layers is the number of layers in the network, excluding the input and output layers.
    The activation function is the SiLU function.
    """

    def __init__(self, input_size, num_layers, embedding_size):
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer greater than 0")
        if not isinstance(embedding_size, int) or embedding_size < 1:
            raise ValueError("embedding_size must be an integer greater than 0")
        super(VariableNet, self).__init__()
        self.act = nn.SiLU()

        self.layers = nn.Sequential()
        self.layers.add_module("fc0", nn.Linear(input_size, embedding_size))
        for i in range(1, num_layers + 1):
            self.layers.add_module(f"act{i}", self.act)
            self.layers.add_module(
                f"fc{i+1}", nn.Linear(embedding_size, embedding_size)
            )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single tensor.
        Can be used after training to get the embedding of a tensor.
        :param x: A tensor.
        :return: The embedding of the tensor.
        """
        x = self.layers(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(
        self, triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of a triplet.
        :param triplet: A tuple of (ancor, positive, negative) tensors.
        :return: A tuple of (ancor, positive, negative) embeddings.
        """
        ancor, positive, negative = triplet[0], triplet[1], triplet[2]
        output1 = self.forward_once(ancor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)
        return output1, output2, output3


def get_embeddings(model: VariableNet | FamusNet, arr: np.ndarray) -> np.ndarray:
    """
    Get the embeddings of an array of tensors.
    :param model: The model to use.
    :param arr: A matrix of samples x features.
    :return: An array of embeddings.
    """
    model.eval()
    tensor_input = torch.tensor(arr)
    with torch.no_grad():
        embeddings = model.module.forward_once(tensor_input)
    return embeddings.numpy()
