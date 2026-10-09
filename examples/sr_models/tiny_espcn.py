"""Minimal SR model that follows the SR4Rec module interface (method 2).

Interface required by SR4Rec:
  * the class is a torch.nn.Module;
  * it has an integer class attribute `scale` (the upscaling factor);
  * __init__ accepts one optional argument `weights` (path or None);
  * forward(x) receives a float32 RGB tensor in [0, 1] with shape (N, 3, H, W)
    and returns a float32 RGB tensor with shape (N, 3, H * scale, W * scale).
Optional: an integer attribute `size_multiple`; SR4Rec then pads the input so
that its height and width are multiples of it, and crops the output back.
SR4Rec calls model.eval(), runs forward() under torch.no_grad(), moves the model
to the configured device, and clamps the output to [0, 1].

The weights file tiny_espcn_x4_pets.pth is produced by train_tiny_espcn.py.
"""


import torch
from torch import nn


class TinyESPCN(nn.Module):
    scale = 4

    def __init__(self, weights: str | None = None):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * self.scale ** 2, 3, padding=1),
            nn.PixelShuffle(self.scale),
        )
        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.load_state_dict(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
