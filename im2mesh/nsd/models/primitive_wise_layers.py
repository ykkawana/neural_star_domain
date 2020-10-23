from torch import nn


class PrimitiveWiseMaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super().__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        """
    Arguments:
      input_data (B, Points, n_primitives, input_channels)
    Returns:
      output_data (B, 1, n_primitives, output_channels)
    """
        assert len(input_data.shape) == 4
        assert input_data.shape[-1] == self.num_channels
        assert input_data.shape[-3] == self.num_points
        B, _, N, _ = input_data.shape
        output_data = input_data.max(1)[0].unsqueeze(1)
        return output_data


class PrimitiveWiseLinear(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_channels,
                 output_channels,
                 act='leaky',
                 bias=True):
        """
    Calculate n_primitive wise feature encoding.
    Weight is separated by n_primitives.
    n_primitive wise calculation is done by group wise convolution.

    Arguments:
      act (str): One of 'leaky', 'relu', 'none'
    """
        super(PrimitiveWiseLinear, self).__init__()
        self.n_primitives = n_primitives
        self.input_channels = input_channels
        self.output_channels = output_channels

        linear = nn.Conv1d(self.n_primitives * self.input_channels,
                           self.n_primitives * self.output_channels,
                           kernel_size=1,
                           groups=self.n_primitives,
                           bias=bias)

        layers = [linear]

        if act == 'leaky':
            layers.append(nn.LeakyReLU(inplace=True))
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, input_data):
        """
    Arguments:
      input_data (B, Points, n_primitives, input_channels)
    Returns:
      output_data (B, Points, n_primitives, output_channels)
    """
        """
        assert len(input_data.shape) == 4
        assert input_data.shape[-1] == self.input_channels
        assert input_data.shape[-2] == self.n_primitives
        B, P, _, _ = input_data.shape
        input_data_BPxNDx1 = input_data.view(
            B * P, self.n_primitives * self.input_channels, 1)
        output_data_BPxNDx1 = self.main(input_data_BPxNDx1)
        output_data = output_data_BPxNDx1.view(B, P, self.n_primitives,
                                               self.output_channels)
        return output_data
        """

        B, N, D, P = input_data.shape
        out = self.main(input_data.view(B, N * D, P))
        return out.view(B, N, self.output_channels, P)
