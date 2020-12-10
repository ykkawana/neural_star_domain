import torch
from torch import nn
from im2mesh.nsd.models import geometry_utils
from im2mesh.nsd.models import radius_decoder
from im2mesh.nsd.models import sphere_decoder

EPS = 1e-7


class NeuralStarDomainDecoder(sphere_decoder.SphereDecoder):
    def __init__(self,
                 num_points,
                 *args,
                 last_scale=.1,
                 factor=1,
                 act='leaky',
                 decoder_class='PrimitiveWiseGroupConvDecoder',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
        self.num_points = num_points
        self.num_labels = 1
        self.theta_dim = 2 if self.dim == 2 else 4
        self.last_scale = last_scale
        self.act = act

        c64 = 64 // self.factor
        self.encoder_dim = c64 * 2

        shaper_decoder_input_dim = self.num_points
        self.decoder = radius_decoder.decoder_dict[decoder_class](
            self.n_primitives,
            shaper_decoder_input_dim,
            self.num_labels,
            dim=self.dim,
            factor=self.factor,
            act=self.act)

    def transform_circumference_angle_to_super_shape_world_cartesian_coord(
        self, thetas, radius, primitive_params, *args, points=None, **kwargs):
        assert not points is None

        assert len(thetas.shape) == 3, thetas.shape
        B, P, D = thetas.shape
        thetas_reshaped = thetas.view(B, 1, P, D)

        assert len(radius.shape) == 4, radius.shape
        # r = (B, n_primitives, P, dim - 1)
        r = radius.view(B, self.n_primitives, P, D)

        coord = geometry_utils.polar2cartesian(r, thetas_reshaped)
        assert not torch.isnan(coord).any()
        # posed_coord = B, n_primitives, P, dim
        if self.learn_pose:
            posed_coord = self.project_primitive_to_world(
                coord, primitive_params)
        else:
            posed_coord = coord

        #print('')
        #print('get pr from points')
        periodic_net_r = self.get_periodic_net_r(thetas.unsqueeze(1), points,
                                                 r[..., -1], posed_coord)

        final_r = r.clone()
        #print('mean r1 in points', r[..., -1].mean())
        final_r[..., -1] = r[..., -1] + periodic_net_r.squeeze(-1)
        #print('mean final r in points', final_r[..., -1].mean())

        final_r = final_r.clamp(min=EPS)
        # B, n_primitives, P, dim

        cartesian_coord = geometry_utils.polar2cartesian(
            final_r, thetas_reshaped)

        assert [*cartesian_coord.shape] == [B, self.n_primitives, P, self.dim]
        assert not torch.isnan(cartesian_coord).any()

        if self.learn_pose:
            posed_cartesian_coord = self.project_primitive_to_world(
                cartesian_coord, primitive_params)
        else:
            posed_cartesian_coord = cartesian_coord
        # B, n_primitives, P, dim
        return posed_cartesian_coord

    def get_periodic_net_r(self, thetas, points, radius, coord):
        # B, 1 or N, P, dim - 1
        #print('mean coord in pr', coord.mean())
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert points.shape[0] == thetas.shape[0]

        B = points.shape[0]
        assert points.shape[1] == self.num_points

        _, _, P, _ = thetas.shape
        # B, P, N

        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        assert [*coord.shape] == [B, self.n_primitives, P, self.dim]

        radius = self.decoder(points, thetas, radius, coord)
        radius = radius * self.last_scale

        #print('mean from pr ', radius.mean())
        return radius

    def get_indicator(self,
                      x,
                      y,
                      z,
                      r1,
                      r2,
                      theta,
                      phi,
                      params,
                      *args,
                      points=None,
                      **kwargs):
        assert not points is None
        coord_list = [x, y]
        is3d = len(z.shape) == len(x.shape)
        if is3d:
            # 3D case
            coord_list.append(z)
            angles = torch.stack([theta, phi], axis=-1)
            radius = r2
        else:
            angles = theta.unsqueeze(-1)
            radius = r1
        coord = torch.stack(coord_list, axis=-1)

        posed_coord = geometry_utils.polar2cartesian(
            torch.stack([r1, r2], axis=-1), angles)
        # posed_coord = B, n_primitives, P, dim
        if self.learn_pose:
            posed_coord = self.project_primitive_to_world(posed_coord, params)

        #print('get pr from sgn')
        rp = self.get_periodic_net_r(angles, points, radius, posed_coord)

        #print('mean r1 in sgn', r1.mean())
        numerator = (coord**2).sum(-1)

        if is3d:
            r2 = r2 + rp.squeeze(-1)
            r2 = r2.clamp(min=EPS)
        else:
            r1 = r1 + rp.squeeze(-1)
            r1 = r1.clamp(min=EPS)

        indicator = 1. - (coord**2).sum(-1).clamp(
            min=EPS).sqrt() / r2

        return indicator

    def get_sgn(self, coord, params, *args, **kwargs):
        r1, r2, theta, phi = self.cartesian2polar(coord, params, *args,
                                                      **kwargs)

        dim = coord.shape[-1]
        x = coord[..., 0]
        y = coord[..., 1]
        z = torch.zeros([1], device=coord.device) if dim == 2 else coord[...,
                                                                         2]
        indicator = self.get_indicator(x, y, z, r1, r2, theta, phi, params,
                                       *args, **kwargs)
        assert not torch.isnan(indicator).any(), indicator
        return indicator
