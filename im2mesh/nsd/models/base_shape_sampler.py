import torch
from torch import nn
from im2mesh.nsd.models import geometry_utils
import math

EPS = 1e-7


class BaseShapeSampler(nn.Module):
    def __init__(self,
                 n_primitives,
                 learn_pose=True,
                 linear_scaling=True,
                 disable_learn_pose_but_transition=False,
                 extract_surface_point_by_max=False,
                 dim=2):
        """Intitialize SuperShapeSampler.

        Args:
            n_primitives: Number of primitives to use.
            learn_pose: Apply pose change flag
            linear_scaling: linearly scale the shape flag
            dim: if dim == 2, then 2D mode. If 3, 3D mode.
        """
        super().__init__()
        self.n_primitives = n_primitives
        self.learn_pose = learn_pose
        self.linear_scaling = linear_scaling
        self.disable_learn_pose_but_transition = disable_learn_pose_but_transition
        self.extract_surface_point_by_max = extract_surface_point_by_max

        self.dim = dim
        if not self.dim in [2, 3]:
            raise NotImplementedError('dim must be either 2 or 3.')

    def transform_circumference_angle_to_super_shape_radius(
        self, thetas, primitive_params, *args, **kwargs):
        """
    Arguments:
      thetas (B, P): Angles between 0 to 2pi
      primitive_params (list or set): containing supershape's parameters 
    Return:
      radius (float): radius value of super shape corresponding to theta
    """
        assert len(thetas.shape) == 3
        B = thetas.shape[0]
        P = thetas.shape[1]
        D = thetas.shape[2]
        assert D == self.dim - 1

        # B, 1, P
        thetas_dim_added = thetas.view(B, 1, P, D)
        assert [*thetas_dim_added.shape] == [B, 1, P, D]

        r = self.get_r_check_shape(thetas_dim_added, primitive_params, *args,
                                   **kwargs)
        assert [*r.shape] == [B, self.n_primitives, P, D]
        assert not torch.isnan(r).any()

        # r = (B, n_primitives, P, D)
        return r

    def get_r_check_shape(self, thetas, params, *args, **kwargs):
        B, N, P, D = thetas.shape
        assert N in [1, self.n_primitives]

        r = self.get_r(thetas, params, *args, **kwargs)

        B2, _, P2, D2 = r.shape
        assert [*r.shape] == [B2, self.n_primitives, P2, D2]
        return r

    def transform_circumference_angle_to_super_shape_world_cartesian_coord(
        self, thetas, radius, primitive_params, *args, **kwargs):

        assert len(thetas.shape) == 3, thetas.shape
        B, P, D = thetas.shape

        assert len(radius.shape) == 4, radius.shape
        # r = (B, n_primitives, P, dim - 1)
        r = radius.view(B, self.n_primitives, P, D)

        thetas_reshaped = thetas.view(B, 1, P, D)
        # B, n_primitives, P, dim
        cartesian_coord = geometry_utils.polar2cartesian(
            r, thetas_reshaped)
        assert [*cartesian_coord.shape] == [B, self.n_primitives, P, self.dim]
        assert not torch.isnan(cartesian_coord).any()

        if self.learn_pose:
            posed_cartesian_coord = self.project_primitive_to_world(
                cartesian_coord, primitive_params)
        else:
            posed_cartesian_coord = cartesian_coord

        # B, n_primitives, P, dim
        return posed_cartesian_coord

    def project_primitive_to_world(self, cartesian_coord, params):
        rotation = params['rotation']
        transition = params['transition']
        linear_scale = params['linear_scale']
        B = cartesian_coord.shape[0]
        if self.linear_scaling and not self.disable_learn_pose_but_transition:
            scaled_cartesian_coord = cartesian_coord * linear_scale.view(
                B, self.n_primitives, 1, self.dim)
        else:
            scaled_cartesian_coord = cartesian_coord

        if not self.disable_learn_pose_but_transition:
            rotated_cartesian_coord = geometry_utils.apply_rotation(
                scaled_cartesian_coord, rotation)
        else:
            rotated_cartesian_coord = scaled_cartesian_coord
        posed_cartesian_coord = rotated_cartesian_coord + transition.view(
            B, self.n_primitives, 1, self.dim)
        assert not torch.isnan(posed_cartesian_coord).any()

        return posed_cartesian_coord

    def transform_world_cartesian_coord_to_tsd(self, coord, params, *args,
                                               **kwargs):
        rotation = params['rotation']
        transition = params['transition']
        linear_scale = params['linear_scale']

        assert len(coord.shape) == 3, coord.shape
        B, P, D = coord.shape
        assert D == self.dim
        assert self.dim in [2, 3]

        if self.learn_pose:
            # B, n_primitives, P, 2
            coord_translated = coord.view(B, 1, P, D) - transition.view(
                B, self.n_primitives, 1, D)

            if not self.disable_learn_pose_but_transition:
                rotated_coord = geometry_utils.apply_rotation(coord_translated,
                                                           rotation,
                                                           inv=True)
            else:
                rotated_coord = coord_translated

            if self.linear_scaling and not self.disable_learn_pose_but_transition:
                rotated_coord = rotated_coord / linear_scale.view(
                    B, self.n_primitives, 1, D)

            # B, n_primitives, P, D
            coord_dim_added = rotated_coord.view(B, self.n_primitives, P, D)
        else:
            coord_dim_added = coord.view(B, 1, P,
                                         D).expand(-1, self.n_primitives, -1,
                                                   -1)

        sgn = self.get_sgn(coord_dim_added, params, *args, **kwargs)
        assert [*sgn.shape] == [B, self.n_primitives, P], sgn.shape
        return sgn

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

    def get_indicator(self, x, y, z, r1, r2, theta, phi, *args, **kwargs):
        """get istropic indicator values.

        Args:
            x: x in cartesian coordinate.
            y: y in cartesian coordinate.
            r1: radius in polar coordinate.
            r2: radius in polar coordinate. Defaults to 1.
            z: z in cartesian coordinate. Defaulst to 0.
            theta: polar coordinate of r1
            phi: polar coordinate of r2

        Returns:
            indicator: indicator value. Positive in inside, negative in outside.
        """
        numerator = (x**2. + y**2. + z**2.)
        denominator = ((phi.cos()**2.) * (r1**2. - 1) + 1 + EPS)
        indicator = 1. - (1. /
                          (r2 + EPS)) * (numerator / denominator + EPS).sqrt()
        return indicator

    def cartesian2polar(self, coord, params, *args, **kwargs):
        """Convert polar coordinate to cartesian coordinate.
        Args:
            coord: (B, N, P, D)
        """
        B, _, P, dim = coord.shape
        x = coord[..., 0]
        y = coord[..., 1]
        z = torch.zeros([1], device=coord.device) if dim == 2 else coord[...,
                                                                         2]
        x_non_zero = torch.where(x == 0, EPS + x, x)
        #x_non_zero = torch.where(x == 0, x.sign() * EPS, x)
        theta = torch.atan2(y, x_non_zero)

        assert not torch.isnan(theta).any(), (theta)
        r1 = self.get_r_check_shape(theta.view(B, self.n_primitives, P, 1),
                                    params, *args, **kwargs)[..., 0]

        phi = torch.atan(z * r1 * theta.cos() / x_non_zero)
        #phi = torch.atan(
        #    (x**2 + y**2).sqrt() / torch.where(z == 0, EPS + z, z))

        assert not torch.isnan(phi).any(), (phi)
        r2 = torch.ones_like(r1) if dim == 2 else (self.get_r_check_shape(
            phi.view(B, self.n_primitives, P, 1), params, *args, **
            kwargs))[..., 1]

        # (B, N, P)
        return r1, r2, theta, phi

    def cartesian2sphere(self, coord, params, *args, **kwargs):
        """Convert polar coordinate to cartesian coordinate.
        Args:
            coord: (B, N, P, D)
        """
        dim = coord.shape[-1]
        B, _, P, dim = coord.shape
        x = coord[..., 0]
        y = coord[..., 1]
        z = torch.zeros([1], device=coord.device) if dim == 2 else coord[...,
                                                                         2]
        x_non_zero = torch.where(x == 0, x + EPS, x)
        theta = torch.atan2(y, x_non_zero)

        assert not torch.isnan(theta).any(), (theta)
        #r = (coord**2).sum(-1).clamp(min=EPS).sqrt()
        r = self.get_r_check_shape(theta.view(B, self.n_primitives, P, 1),
                                   params, *args, **kwargs)[..., 0]
        #print('r in c2p', r.mean())
        assert not torch.isnan(r).any(), (r)

        xysq_non_zero = (x_non_zero**2 + y**2).clamp(min=EPS).sqrt()
        #xysq_non_zero = torch.where(xysq == 0, EPS + xysq, xysq)
        #xysq_non_zero = xysq.clamp(min=EPS)
        #phi = torch.atan(z / xysq_non_zero)
        phi = torch.atan2(z, xysq_non_zero)
        #phi = torch.atan(xysq_non_zero / z)
        #phi = torch.acos((z / r_phi))

        assert not torch.isnan(phi).any(), (phi)

        # (B, N, P)
        return r.unsqueeze(-1).expand([*coord.shape[:-1],
                                       dim - 1]), torch.stack([theta, phi],
                                                              axis=-1)

    def extract_super_shapes_surface_point(self, super_shape_point,
                                           primitive_params, *args, **kwargs):
        """Extract surface point for visualziation purpose"""
        assert len(super_shape_point.shape) == 4
        output_sgn_BxNxNP = self.extract_surface_point_std(
            super_shape_point, primitive_params, *args, **kwargs)
        B, N, P, D = super_shape_point.shape
        B2, N2, NP = output_sgn_BxNxNP.shape
        assert B == 1 and B2 == 1, 'only works with batch size 1'
        surface_mask = self.extract_super_shapes_surface_mask(
            output_sgn_BxNxNP, *args, **kwargs)
        return super_shape_point[surface_mask].view(1, -1, D)

    def extract_super_shapes_surface_mask(self, output_sgn_BxNxNP, *args,
                                          **kwargs):
        B, N, NP = output_sgn_BxNxNP.shape
        P = NP // N
        output_sgn = output_sgn_BxNxNP.view(B, self.n_primitives,
                                            self.n_primitives, P)
        if self.extract_surface_point_by_max:
            sgn_p_BxPsN = torch.relu(output_sgn.max(1)[0]).view(
                B, self.n_primitives, P)
            surface_mask = (sgn_p_BxPsN <= 1e-4)
        else:
            sgn_p_BxPsN = nn.functional.relu(output_sgn).sum(1).view(
                B, self.n_primitives, P)
            surface_mask = (sgn_p_BxPsN <= 1e-1)
        return surface_mask

    def extract_surface_point_std(self, super_shape_point, primitive_params,
                                  *args, **kwargs):
        B, N, P, D = super_shape_point.shape

        super_shape_coord = super_shape_point.view(B, N * P, D)

        # B, N, N * P
        output_sgn_BxNxNP = self.transform_world_cartesian_coord_to_tsd(
            super_shape_coord, primitive_params, *args, **kwargs)
        assert [*output_sgn_BxNxNP.shape] == [B, N, N * P]
        return output_sgn_BxNxNP

    def generate_surface_mask(self, super_shape_point, primitive_params, *args,
                              **kwargs):
        B, N, P, D = super_shape_point.shape

        def gen_surface_mask(idx):
            super_shape_point_p = super_shape_point[:, idx, :, :]
            # B, N, P
            output_sgn = self.transform_world_cartesian_coord_to_tsd(
                super_shape_point_p, primitive_params, *args, **kwargs)
            sgn_p_BxPs = nn.functional.relu(output_sgn).sum(1).view(B, 1, P)
            surface_mask = (sgn_p_BxPs <= 1e-1)
            output_sgn_self = torch.relu(output_sgn[:, idx, :].view(B, 1, P) -
                                         1.2)
            return surface_mask, output_sgn_self

        result = [gen_surface_mask(idx) for idx in range(N)]
        surface_mask_list = [res[0] for res in result]
        surface_mask = torch.cat(surface_mask_list, axis=1)
        self_sgn_list = [res[1] for res in result]
        self_sgn = torch.cat(self_sgn_list, axis=1)
        return surface_mask, self_sgn

    def forward(self,
                primitive_params,
                *args,
                thetas=None,
                coord=None,
                return_surface_mask=True,
                **kwargs):
        if thetas is not None:
            # B, N, P
            radius = self.transform_circumference_angle_to_super_shape_radius(
                thetas, primitive_params, *args, **kwargs)
            # B, N, P, dim
            super_shape_point = self.transform_circumference_angle_to_super_shape_world_cartesian_coord(
                thetas, radius, primitive_params, *args, **kwargs)

            if return_surface_mask:
                output_sgn_BxNxNP = self.extract_surface_point_std(
                    super_shape_point, primitive_params, *args, **kwargs)

                # B, P', dim
                surface_mask = self.extract_super_shapes_surface_mask(
                    output_sgn_BxNxNP, *args, **kwargs)
                #surface_mask, output_sgn_BxNxNP = self.generate_surface_mask(
                #    super_shape_point, primitive_params, *args, **kwargs)
            else:
                surface_mask = None
                output_sgn_BxNxNP = None
        else:
            super_shape_point = None
            surface_mask = None
            output_sgn_BxNxNP = None

        if coord is not None:
            tsd = self.transform_world_cartesian_coord_to_tsd(
                coord, primitive_params, *args, **kwargs)
        else:
            tsd = None

        # (B, N, P, dim), (B, N, P), (B, N, P2), (B, N, NxP)
        return super_shape_point, surface_mask, tsd, output_sgn_BxNxNP
