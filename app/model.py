from typing import Tuple

import torch
from torch import nn
import os

torch.set_default_dtype(torch.float)


class MLP(nn.Module):
    def __init__(
        self,
        profile_directory_path=None,
        input_size=None,
        num_layers=3,
        embedding_size=320,
    ):
        if profile_directory_path is None and input_size is None:
            raise ValueError(
                "Either profile_directory_path or input_size must be provided"
            )
        if profile_directory_path is not None and input_size is not None:
            raise ValueError(
                "Only one of profile_directory_path or input_size can be provided"
            )
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer greater than 0")
        if not isinstance(embedding_size, int) or embedding_size < 1:
            raise ValueError("embedding_size must be an integer greater than 0")

        if profile_directory_path is not None:
            input_size = len(os.listdir(profile_directory_path))
        super(MLP, self).__init__()
        self.act = nn.SiLU()

        self.layers = nn.Sequential()
        self.layers.add_module("fc0", nn.Linear(input_size, embedding_size))
        for i in range(1, num_layers + 1):
            self.layers.add_module(f"act{i}", self.act)
            self.layers.add_module(
                f"fc{i + 1}", nn.Linear(embedding_size, embedding_size)
            )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(
        self, triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ancor, positive, negative = triplet[0], triplet[1], triplet[2]
        output1 = self.forward_once(ancor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)
        return output1, output2, output3

    def save_state(self, path):
        torch.save(self.state_dict(), path)


def load_from_state(state_path):
    state = torch.load(state_path)
    input_size = state["layers.fc0.weight"].shape[1]
    num_inner_layers = len([k for k in state.keys() if "weight" in k]) - 1
    model = MLP(input_size=input_size, num_layers=num_inner_layers, embedding_size=320)
    model.load_state_dict(state)
    return model
