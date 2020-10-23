import torch.nn as nn
import torch.nn.functional as F
import torch
from im2mesh.nsd.models import transformation_decoder, nsd_sampler
import time

EPS = 1e-7


class PeriodicShapeDecoderSimplest(nn.Module):
    ''' Decoder with CBN class 2.

    It differs from the previous one in that the number of blocks can be
    chosen.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''
    def __init__(
        self,
        dim=3,
        z_dim=0,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        max_m=4,
        n_primitives=6,
        shape_sampler_decoder_factor=1,
        is_shape_sampler_sphere=False,
        transition_range=3.,
        paramnet_class='ParamNet',
        paramnet_hidden_size=128,
        paramnet_dense=True,
        is_single_paramnet=False,
        layer_depth=0,
        skip_position=3,  # count start from input fc
        is_skip=True,
        shape_sampler_decoder_class='PrimitiveWiseGroupConvDecoder',
        disable_learn_pose_but_transition=False,
        freeze_primitive=False,
        no_last_bias=False,
        supershape_freeze_rotation_scale=False,
        get_features_from=[],
        concat_input_feature_with_pose_feature=False,
        spherical_angles=False,
        is_simpler_sgn=False,
        extract_surface_point_by_max=False,
        last_scale=.1):
        super().__init__()
        assert dim in [2, 3]
        self.get_features_from = get_features_from
        self.concat_input_feature_with_pose_feature = concat_input_feature_with_pose_feature

        self.primitive = transformation_decoder.TransformationDecoder(
            max_m,
            n_primitives,
            latent_dim=c_dim,
            train_logits=False,
            transition_range=transition_range,
            paramnet_class=paramnet_class,
            paramnet_hidden_size=paramnet_hidden_size,
            paramnet_dense=paramnet_dense,
            is_single_paramnet=is_single_paramnet,
            layer_depth=layer_depth,
            is_skip=is_skip,
            skip_position=skip_position,
            dim=dim,
            supershape_freeze_rotation_scale=supershape_freeze_rotation_scale,
            get_features_from=get_features_from)
        if freeze_primitive:
            self.primitive.require_grad = False

        if get_features_from:
            if concat_input_feature_with_pose_feature:
                feature_dim = paramnet_hidden_size + c_dim
            else:
                feature_dim = paramnet_hidden_size
        else:
            feature_dim = c_dim
        self.p_sampler = nsd_sampler.PeriodicShapeSampler(
            feature_dim,
            max_m,
            n_primitives,
            dim=dim,
            factor=shape_sampler_decoder_factor,
            last_scale=last_scale,
            disable_learn_pose_but_transition=disable_learn_pose_but_transition,
            decoder_class=shape_sampler_decoder_class,
            no_last_bias=no_last_bias,
            is_simpler_sgn=is_simpler_sgn,
            extract_surface_point_by_max=extract_surface_point_by_max)

    def forward(self,
                coord,
                _,
                color_feature,
                angles=None,
                only_return_points=False,
                **kwargs):
        params = self.primitive(color_feature)

        if self.get_features_from:
            if len(self.get_features_from) == 1:
                feature = params[self.get_features_from[0] + '_feature']

                if self.concat_input_feature_with_pose_feature:
                    feature = torch.cat([feature, color_feature], axis=-1)
            else:
                raise NotImplementedError
        else:
            feature = color_feature

        if only_return_points:
            output = self.p_sampler(params,
                                    thetas=angles,
                                    coord=None,
                                    points=feature,
                                    return_surface_mask=False)
        else:
            output = self.p_sampler(params,
                                    thetas=angles,
                                    coord=coord,
                                    points=feature,
                                    return_surface_mask=True)
        pcoord, o1, o2, o3 = output
        output = (pcoord, o1, o2, o3, None)

        return output

    def to(self, device):
        super().to(device)
        self.p_sampler.to(device)
        self.primitive.to(device)
        return self
