from famus.logging import logger
from typing import List

try:
    import torch

    torch.set_default_dtype(torch.float)
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class nn:
        class Module:
            def __init__(self):
                pass

    logger.warning(
        "PyTorch is not installed. Please install PyTorch to use the model module."
    )


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "\n" + "=" * 70 + "\nPyTorch is required to use FAMUS models!\n\n"
        )


class MLP(nn.Module):
    _require_torch()

    def __init__(
        self, input_dim: int, embedding_dim: int = 128, hidden_dims: List[int] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [320] * 3

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        embedding = self.network(x)
        # L2 normalization can help stabilize training
        return nn.functional.normalize(embedding, p=2, dim=1)

    @staticmethod
    def load_from_state(state_path, device=None):
        state = torch.load(state_path, map_location=device, weights_only=False)
        input_dim = state["model_state_dict"]["network.0.weight"].size(1)
        embedding_dim = 320

        model = MLP(input_dim, embedding_dim)
        model.load_state_dict(state["model_state_dict"])

        if device:
            model.to(device)

        return model
