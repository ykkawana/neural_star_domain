import torch
from torch import nn
import math
from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                            CBatchNorm1d_legacy, ResnetBlockConv1d)


class TransformationDecoder(nn.Module):
    def __init__(
        self,
        n_primitives,
        latent_dim=100,
        train_logits=True,
        use_transformation_decoder=True,
        transformation_decoder_dense=True,
        transition_range=1.,
        transformation_decoder_class='ParamNet',
        transformation_decoder_hidden_size=128,
        is_single_transformation_decoder=False,
        layer_depth=0,
        skip_position=3,  # count start from input fc
        is_skip=True,
        supershape_freeze_rotation_scale=False,
        get_features_from=[],
        dim=2):
        """Initialize SuperShapes.

        Args:
            max_m: Max number of vertices in a primitive. If quadrics is True,
                then max_m must be larger than 4.
            n_primitives: Number of primitives to use.
            train_logits: train logits flag.
            quadrics: quadrics mode flag. In this mode, n1, n2, n3 are set
                to one and not trained, and logits is not trained and 5th 
                element of m_vector is always one and the rest are zero.
            dim: if dim == 2, then 2D mode. If 3, 3D mode.
        Raises:
            NotImplementedError: when dim is neither 2 nor 3.
        """
        super().__init__()
        self.n_primitives = n_primitives
        self.train_logits = train_logits
        self.dim = dim
        self.transition_range = transition_range
        self.latent_dim = latent_dim
        self.use_transformation_decoder = use_transformation_decoder
        self.is_single_transformation_decoder = is_single_transformation_decoder
        self.get_features_from = get_features_from

        if get_features_from:
            assert self.use_transformation_decoder

        if not self.dim in [2, 3]:
            raise NotImplementedError('dim must be either 2 or 3.')

        self.rot_dim = 1 if self.dim == 2 else 4  # 1 for euler angle for 2D, 4 for quaternion for 3D
        if self.is_single_transformation_decoder:
            assert not supershape_freeze_rotation_scale, 'supershape_freeze_rotation_scale doesnt support single transformation_decoder'
            self.transformation_decoder = transformation_decoder_dict[transformation_decoder_class](
                self.n_primitives,
                self.latent_dim,
                self.dim + self.rot_dim + self.dim + 1,
                hidden_size=transformation_decoder_hidden_size,
                layer_depth=layer_depth,
                dense=transformation_decoder_dense)
        else:
            self.transition_net = transformation_decoder_dict[transformation_decoder_class](
                self.n_primitives,
                self.latent_dim,
                self.dim,
                hidden_size=transformation_decoder_hidden_size,
                layer_depth=layer_depth,
                dense=transformation_decoder_dense)
            self.rotation_net = transformation_decoder_dict[transformation_decoder_class](
                self.n_primitives,
                self.latent_dim,
                self.rot_dim,
                hidden_size=transformation_decoder_hidden_size,
                layer_depth=layer_depth,
                dense=transformation_decoder_dense)
            self.scale_net = transformation_decoder_dict[transformation_decoder_class](
                self.n_primitives,
                self.latent_dim,
                self.dim,
                hidden_size=transformation_decoder_hidden_size,
                layer_depth=layer_depth,
                dense=transformation_decoder_dense)
            if supershape_freeze_rotation_scale:
                self.scale_net.requires_grad = False
                self.rotation_net.requires_grad = False

            self.prob_net = transformation_decoder_dict[transformation_decoder_class](
                self.n_primitives,
                self.latent_dim,
                1,
                hidden_size=transformation_decoder_hidden_size,
                layer_depth=layer_depth,
                dense=transformation_decoder_dense)

        # Pose params
        # B=1, n_primitives, 2
        self.transition = nn.Parameter(torch.Tensor(1, n_primitives, self.dim))

        self.rotation = nn.Parameter(
            torch.Tensor(1, n_primitives, self.rot_dim))

        # B=1, n_primitives, 2
        self.linear_scale = nn.Parameter(
            torch.Tensor(1, n_primitives, self.dim))

        # B=1, n_primitives
        self.prob = nn.Parameter(torch.Tensor(1, n_primitives, 1))

        if self.use_transformation_decoder:
            self.linear_scale.requires_grad = False
            self.transition.requires_grad = False
            self.rotation.requires_grad = False
            self.prob.requires_grad = False

        self.weight_init()

    def weight_init(self):
        torch.nn.init.uniform_(self.rotation, 0, 1)
        torch.nn.init.uniform_(self.linear_scale, 0, 1)
        torch.nn.init.uniform_(self.transition, -self.transition_range,
                               self.transition_range)
        torch.nn.init.uniform_(self.prob, 0, 1)

    def get_primitive_params(self, x):
        B = x.shape[0]

        params = {}

        if self.use_transformation_decoder and self.is_single_transformation_decoder:
            pose_param = self.transformation_decoder(x)

            pointer = 0
            next_pointer = self.rot_dim
            rotation_param = pose_param[:, :, pointer:next_pointer]

            pointer = next_pointer
            next_pointer = pointer + self.dim
            transition_param = pose_param[:, :, pointer:next_pointer]

            pointer = next_pointer
            next_pointer = pointer + self.dim
            scale_param = pose_param[:, :, pointer:next_pointer]

            pointer = next_pointer
            next_pointer = pointer + 1
            prob_param = pose_param[:, :, pointer:next_pointer]

        if self.use_transformation_decoder and not self.is_single_transformation_decoder:
            rotation = self.rotation + self.rotation_net(x)
        elif self.use_transformation_decoder and self.is_single_transformation_decoder:
            rotation = self.rotation + rotation_param
        else:
            rotation = self.rotation.repeat(B, 1, 1)
        if self.dim == 2:
            rotation = torch.tanh(self.rotation) * math.pi
        else:
            rotation = nn.functional.normalize(rotation, dim=-1)

        if self.use_transformation_decoder and not self.is_single_transformation_decoder:
            transition = self.transition + self.transition_net(x)
        elif self.use_transformation_decoder and self.is_single_transformation_decoder:
            transition = self.transition + transition_param
        else:
            transition = self.transition.repeat(B, 1, 1)

        if self.use_transformation_decoder and not self.is_single_transformation_decoder:
            linear_scale = self.linear_scale + self.scale_net(x)
        elif self.use_transformation_decoder and self.is_single_transformation_decoder:
            linear_scale = self.linear_scale + scale_param
        else:
            linear_scale = self.linear_scale.repeat(B, 1, 1)
        linear_scale = torch.tanh(linear_scale) + 1.1

        if self.use_transformation_decoder and not self.is_single_transformation_decoder:
            prob = self.prob + self.prob_net(x)
        elif self.use_transformation_decoder and self.is_single_transformation_decoder:
            prob = self.prob + prob_param
        else:
            prob = self.prob.repeat(B, 1, 1)
        prob = torch.sigmoid(prob)

        params.update({
            'rotation': rotation,
            'transition': transition,
            'linear_scale': linear_scale,
            'prob': prob
        })

        features = {}
        for feature_name in self.get_features_from:
            features[feature_name + '_feature'] = getattr(
                self, feature_name + '_net').get_feature(x)
        params.update(features)

        return params

    def forward(self, x):
        assert x.ndim == 2  # B, latent dim
        assert x.shape[-1] == self.latent_dim
        return self.get_primitive_params(x)

class ParamNet(nn.Module):
    def __init__(self,
                 n_primitives,
                 in_channel,
                 param_dim,
                 dense=True,
                 layer_depth=0,
                 hidden_size=128,
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.dense = dense
        if layer_depth == 0:
            self.conv1d = nn.Linear(in_channel, in_channel)
            self.out_conv1d = nn.Linear(in_channel, n_primitives * param_dim)
            self.convs = []
        else:
            self.conv1d = nn.Linear(in_channel, hidden_size)
            self.convs = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(layer_depth)
            ])
            self.out_conv1d = nn.Linear(hidden_size, n_primitives * param_dim)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)
        #self.out_conv1d = nn.Conv1d(in_channel, n_primitives * param_dim, 1)

    def get_feature(self, x):
        if self.dense:
            x = self.act(self.conv1d(x))

        for conv in self.convs:
            x = self.act(conv(x))

        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.get_feature(x)

        return self.out_conv1d(x).view(B, self.n_primitives, self.param_dim)


class ParamNetResNetBlock(nn.Module):
    def __init__(self,
                 n_primitives,
                 in_channel,
                 param_dim,
                 hidden_size=128,
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Conv1d(hidden_size, n_primitives * param_dim, 1)
        self.bn = nn.BatchNorm1d(hidden_size)

        self.block0 = ResnetBlockConv1d(hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        net = x.view(B, -1, 1)

        net = self.conv1d(net)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)

        net = self.out_conv1d(self.act(self.bn(net))).view(
            B, self.n_primitives, self.param_dim)

        return net


class ParamNetMLP(nn.Module):
    def __init__(
        self,
        n_primitives,
        in_channel,
        param_dim,
        layer_depth=4,
        skip_position=3,  # count start from input fc
        is_skip=True,
        hidden_size=128,
        **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.in_conv1d = nn.Linear(in_channel, hidden_size)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Linear(hidden_size, n_primitives * param_dim)
        self.layer_depth = layer_depth
        self.is_skip = is_skip
        self.skip_position = skip_position

        self.conv1d = nn.Linear(hidden_size, hidden_size)
        self.conv1d_for_skip = nn.Linear(hidden_size + in_channel, hidden_size)

    def forward(self, x):
        B = x.shape[0]
        reshaped_x = x.view(B, -1)

        net = self.in_conv1d(reshaped_x)

        for idx in range(self.layer_depth):
            if idx == self.skip_position - 1:
                net = torch.cat([net, reshaped_x], axis=1)
                net = self.conv1d_for_skip(net)
            else:
                net = self.conv1d(net)

        net = self.out_conv1d(self.act(net)).view(B, self.n_primitives,
                                                  self.param_dim)

        return net


class ParamNetResNetBlockWOBatchNorm(nn.Module):
    def __init__(self,
                 n_primitives,
                 in_channel,
                 param_dim,
                 hidden_size=128,
                 layer_depth=4,
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.in_conv1d = nn.Linear(in_channel, hidden_size)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Linear(hidden_size, n_primitives * param_dim)
        self.layer_depth = layer_depth

        self.block = ResnetBlockFC(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        net = x.view(B, -1)

        net = self.in_conv1d(net)

        for _ in range(self.layer_depth):
            net = self.block(net)

        net = self.out_conv1d(self.act(net)).view(B, self.n_primitives,
                                                  self.param_dim)

        return net


transformation_decoder_dict = {
    'ParamNet': ParamNet,
    'ParamNetResNetBlock': ParamNetResNetBlock,
    'ParamNetMLP': ParamNetMLP,
    'ParamNetResNetBlockWOBatchNorm': ParamNetResNetBlockWOBatchNorm
}
