import torch

EPS = 1e-7


def polar2cartesian(radius, angles):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        r: radius (B, N, P, 1 or 2) last dim is one in 2D mode, two in 3D.
        angles: angle (B, 1, P, 1 or 2) 
    """
    #print('radius angles in s2c mean', radius[..., -1].mean(), angles.mean())
    dim = radius.shape[-1]
    dim2 = angles.shape[-1]
    P = radius.shape[-2]
    P2 = angles.shape[-2]
    assert dim == dim2
    assert dim in [1, 2]
    assert P == P2

    theta = angles[..., 0]
    phi = torch.zeros([1], device=angles.device) if dim == 1 else angles[...,
                                                                         1]
    r1 = radius[..., 0]
    r2 = torch.ones([1], device=radius.device) if dim == 1 else radius[..., 1]

    phicosr2 = phi.cos() * r2
    cartesian_coord_list = [
        theta.cos() * r1 * phicosr2,
        theta.sin() * r1 * phicosr2
    ]

    # 3D
    if dim == 2:
        cartesian_coord_list.append(phi.sin() * r2)
    return torch.stack(cartesian_coord_list, axis=-1)


def sphere2cartesian(radius, angles):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        r: radius (B, N, P, 1 or 2) last dim is one in 2D mode, two in 3D.
        angles: angle (B, 1, P, 1 or 2) 
    """
    #print('radius angles in s2c mean', radius[..., 0].mean(), angles.mean())
    dim = radius.shape[-1]
    dim2 = angles.shape[-1]
    P = radius.shape[-2]
    P2 = angles.shape[-2]
    assert dim2 in [1, 2]
    assert P == P2

    theta = angles[..., 0]
    phi = torch.zeros([1], device=angles.device) if dim == 1 else angles[...,
                                                                         1]
    r = radius[..., 0]

    phicosr2 = phi.cos() * r
    cartesian_coord_list = [
        theta.cos() * r * phicosr2,
        theta.sin() * r * phicosr2
    ]

    # 3D
    if dim == 2:
        cartesian_coord_list.append(phi.sin() * r)
    return torch.stack(cartesian_coord_list, axis=-1)


def cartesian2sphere(coord):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        coord: (B, N, P, D)
    """
    dim = coord.shape[-1]
    x = coord[..., 0]
    y = coord[..., 1]
    z = torch.zeros([1], device=coord.device) if dim == 2 else coord[..., 2]
    x_non_zero = torch.where(x == 0, x + EPS, x)
    theta = torch.atan2(y, x_non_zero)

    assert not torch.isnan(theta).any(), (theta)
    r = (coord**2).sum(-1).clamp(min=EPS).sqrt()
    assert not torch.isnan(r).any(), (r)

    xysq_non_zero = (x**2 + y**2).clamp(min=EPS).sqrt()
    phi = torch.atan2(z, xysq_non_zero)

    assert not torch.isnan(phi).any(), (phi)

    # (B, N, P)
    return r.unsqueeze(-1).expand([*coord.shape[:-1],
                                   dim - 1]), torch.stack([theta, phi],
                                                          axis=-1)
def apply_rotation(coord, rotation, inv=False):
    B, N, P, dim = coord.shape
    _, N2, dim2 = rotation.shape
    assert N == N2, (N, N2)
    if dim == 2:
        assert dim2 == 1
        return apply_2d_rotation(coord, rotation, inv=inv)
    elif dim == 3:
        assert dim2 == 4
        return apply_3d_rotation(coord, rotation, inv=inv)
    else:
        raise NotImplementedError


def apply_2d_rotation(xy, rotation, inv=False):
    B, N, P, dim = xy.shape
    # B, n_primitives, P, 2, 2
    rotation_matrix = get_2d_rotation_matrix(rotation, inv=inv).view(
        B, N, 1, 2, 2).repeat(1, 1, P, 1, 1)
    assert not torch.isnan(rotation_matrix).any()
    # B, n_primitives, P, 2, 1
    xy_before_rotation = xy.view(B, N, P, 2, 1)
    rotated_xy = torch.bmm(rotation_matrix.view(-1, 2, 2),
                           xy_before_rotation.view(-1, 2, 1)).view(B, N, P, 2)
    return rotated_xy


def apply_3d_rotation(coord, rotation, inv=False):
    B, N, P, dim = coord.shape
    B2, N2, D = rotation.shape
    assert B == B2
    assert N == N2, (N, N2)
    # rotation quaternion in [w, x, y, z]

    q = rotation
    if inv:
        w, x, y, z = q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    qq = rotation**2
    w2, x2, y2, z2 = qq[..., 0], qq[..., 1], qq[..., 2], qq[..., 3]
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                          dim=-1).view(B2, N2, 1, 3, 3)

    rotated_coord = (coord.unsqueeze(-2) * rot_mat).sum(-1)
    return rotated_coord

def get_2d_rotation_matrix(rotation, inv=False):
    sgn = -1. if inv else 1.
    upper_rotation_matrix = torch.cat([rotation.cos(), -sgn * rotation.sin()],
                                      axis=-1)
    lower_rotation_matrix = torch.cat(
        [sgn * rotation.sin(), rotation.cos()], axis=-1)
    rotation_matrix = torch.stack(
        [upper_rotation_matrix, lower_rotation_matrix], axis=-2)
    return rotation_matrix

