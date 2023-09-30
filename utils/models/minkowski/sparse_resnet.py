# Reference: https://github.com/mit-han-lab/torchsparse/blob/master/torchsparse/backbones/resnet.py

__all__ = ['SparseResNet21D']

from torch.nn import Sequential

from utils.models.minkowski.common import SparseConvBNReLU, SparseResidualBlock


class SparseResNet(Sequential):

  def __init__(
      self,
      blocks,
      *,
      in_channels: int = 4,
      width_multiplier: float = 1.0,
  ) -> None:
    super().__init__()
    self.blocks = blocks
    self.in_channels = in_channels
    self.width_multiplier = width_multiplier

    for num_blocks, out_channels, kernel_size, stride in blocks:
      out_channels = int(out_channels * width_multiplier)
      blocks = []
      for index in range(num_blocks):
        if index == 0:
          blocks.append(
              SparseConvBNReLU(
                  ndim=3,
                  in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
              ))
        else:
          blocks.append(
              SparseResidualBlock(
                  ndim=3,
                  in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
              ))
        in_channels = out_channels
      self.append(Sequential(*blocks))

  def forward(self, x):
    outputs = []
    for layer in self:
      x = layer(x)
      outputs.append(x)
    return outputs


class SparseResNet21D(SparseResNet):

  def __init__(self, **kwargs) -> None:
    super().__init__(
        blocks=[
            (3, 16, 3, 1),
            (3, 32, 3, 2),
            (3, 64, 3, 2),
            (3, 128, 3, 2),
            (1, 128, (1, 3, 1), (1, 2, 1)),
        ],
        **kwargs,
    )
