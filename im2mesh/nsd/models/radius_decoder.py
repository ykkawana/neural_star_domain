import torch
from torch import nn
from im2mesh.nsd.models import primitive_wise_layers
import torch.nn.functional as F
import numpy as np


class PrimitiveWiseGroupConvDecoder(nn.Module):
    def __init__(self,
                 n_primitives,
                 input_dim,
                 output_dim,
                 factor=1,
                 dim=3,
                 act='leaky'):
        super().__init__()
        self.n_primitives = n_primitives
        self.output_dim = output_dim
        self.act = act
        self.dim = dim
        c64 = 64 // factor

        self.periodic_input_dim = self.dim #(self.dim - 1) * 2 + self.dim

        self.input_dim = input_dim + self.periodic_input_dim

        self.decoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      self.input_dim,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      self.output_dim,
                                                      act='none'))

    def forward(self, feature, thetas, radius, coord, *args, **kwargs):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert feature.shape[0] == thetas.shape[0]
        B, N, P, Dn1 = thetas.shape
        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        _, feature_dim = feature.shape
        # B, 1 or N, dim - 1, P
        thetas_transposed = thetas.transpose(2, 3).contiguous()

        radius_transposed = radius.view(B, self.n_primitives, 1,
                                        P).contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, N, dim - 1, P
        sincos = (torch.cat([sin, cos], axis=-2) * 100).expand(
            -1, self.n_primitives, -1, -1)

        # Feature dims has to be (B, P, self.n_primitives, feature dim)

        periodic_feature_list = [
            feature.view(B, 1, feature_dim, 1).expand(-1, self.n_primitives,
                                                      -1, P)
        ]

        transposed_target = coord.transpose(2, 3)  #.contiguous()
        periodic_feature_list.append(transposed_target)

        encoded_sincos = torch.cat(periodic_feature_list, axis=-2)
        radius = self.decoder(encoded_sincos).view(
            B, self.n_primitives, self.output_dim,
            P).transpose(2, 3).contiguous()


        return radius

decoder_dict = {
    'PrimitiveWiseGroupConvDecoder': PrimitiveWiseGroupConvDecoder,
}
