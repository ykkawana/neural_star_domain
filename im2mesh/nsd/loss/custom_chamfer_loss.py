import torch
from pykeops.torch import LazyTensor
import numpy as np
from torch.nn import functional as F


def custom_chamfer_loss(primitive_points,
                        target_points,
                        source_normals=None,
                        target_normals=None,
                        prob=None,
                        surface_mask=None,
                        pykeops=True,
                        p='original',
                        apply_surface_mask_before_chamfer=False):
    """
    N = n_primitives
    B = batch
    Ps = points per primitive
    Pt = target points

    Args:
        primitive_points: B x N x Ps x 2
        target_points: B x Pt x 2
        prob: B x N
        surface_mask: B x N x Ps
        p (str): original = (x.abs ** 2).sum() ** 0.5 or euclidean
    """
    assert len(primitive_points.shape) == 4
    assert len(target_points.shape) == 3

    B, N, Ps, _ = primitive_points.shape
    _, Pt, D = target_points.shape

    if p == 'original':
        torch_p = 2
    elif p == 'euclidean':
        torch_p = None
    else:
        raise NotImplementedError

    is_normal_loss = False
    if source_normals is None or target_normals is None:
        normal_loss = None
    else:
        is_normal_loss = True

    if pykeops:
        # B x (Ps x N) x dim
        primitive_points_reshaped = primitive_points.view(B, -1, D)

        if apply_surface_mask_before_chamfer:
            pass

        # B x (Ps * N) x 1 x dim
        primitive_points_BxPsNx1x2 = primitive_points_reshaped.unsqueeze(2)

        # B x 1x Pt x dim
        target_points_Bx1xPtx2 = target_points.unsqueeze(1)

        G_i1 = LazyTensor(primitive_points_BxPsNx1x2)
        X_j1 = LazyTensor(target_points_Bx1xPtx2)

        # B x (Ps * N) x Pt
        if p == 'original':
            dist = ((G_i1 - X_j1).abs()**2).sum(3)**(1. / 2.)
        elif p == 'euclidean':
            dist = (G_i1 - X_j1).norm2()
        else:
            raise NotImplementedError

        # B x (Ps * N)
        idx1 = dist.argmin(dim=2)
        target_points_selected = batched_index_select(target_points, 1,
                                                      idx1.view(B, -1))
        diff_primitives2target = primitive_points_reshaped - target_points_selected
        loss_primitives2target = torch.norm(diff_primitives2target,
                                            torch_p,
                                            dim=2).squeeze(-1)

        idx2 = dist.argmin(dim=1)  # Grid

        primitive_points_selected = batched_index_select(
            primitive_points_reshaped, 1, idx2.view(B, -1))
        diff_target2primitives = primitive_points_selected - target_points
        loss_target2primitives = torch.norm(diff_target2primitives,
                                            torch_p,
                                            dim=2).squeeze(-1).mean()

        if is_normal_loss:
            primitive_normals_reshaped = source_normals.view(B, -1, D)

            target_normals_selected = batched_index_select(
                target_normals, 1, idx1.view(B, -1))
            normal_loss_primitives2target = 1 - torch.abs(
                F.cosine_similarity(primitive_normals_reshaped,
                                    target_normals_selected,
                                    dim=2,
                                    eps=1e-6))

            primitive_normals_selected = batched_index_select(
                primitive_normals_reshaped, 1, idx2.view(B, -1))
            normal_loss_target2primitives = 1 - torch.abs(
                F.cosine_similarity(target_normals,
                                    primitive_normals_selected,
                                    dim=2,
                                    eps=1e-6)).mean()

    else:
        if apply_surface_mask_before_chamfer:
            surface_mask_reshaped = surface_mask.view(-1)
            primitive_points = primitive_points.view(
                -1, D)[surface_mask_reshaped == 1, :].view(B, -1, D)

        # B x 1 x (Ps * N) x 2
        primitive_points_Bx1xPsNx2 = primitive_points.view(B, -1,
                                                           D).unsqueeze(1)

        # B x Pt x 1 x 2
        target_points_BxPtx1x2 = target_points.unsqueeze(2)

        # B x Pt x (Ps * N) x 2
        diff_target2primitives = target_points_BxPtx1x2 - primitive_points_Bx1xPsNx2

        # B x Pt x (Ps * N)
        dist_target2primitives = torch.norm(diff_target2primitives,
                                            torch_p,
                                            dim=3)

        # B x Pt
        loss_target2primitives = torch.min(dist_target2primitives,
                                           dim=2)[0].mean()

        # B x (Ps * N) x 1 x 2
        primitive_points_BxPsNx1x2 = primitive_points.view(B, -1,
                                                           D).unsqueeze(2)

        # B x 1x Pt x 2
        target_points_Bx1xPtx2 = target_points.unsqueeze(1)

        # B x (Ps * N) x Pt x 2
        diff_primitives2target = primitive_points_BxPsNx1x2 - target_points_Bx1xPtx2

        # B x (Ps * N) x Pt
        dist_primitives2target = torch.norm(diff_primitives2target,
                                            torch_p,
                                            dim=3)

        # B x (Ps * N)
        loss_primitives2target = torch.min(dist_primitives2target, dim=2)[0]

    if prob is not None:
        assert len(prob.shape) == 2
        # B x N x Ps
        prob_BxNxPs = prob.view(1, -1, 1).repeat(1, 1, Ps)

        # B x (Ps * N)
        prob_BxPsN = prob_BxNxPs.view(1, -1)

        loss_primitives2target = loss_primitives2target * prob_BxPsN

    if surface_mask is not None and not apply_surface_mask_before_chamfer:
        # B, N, Ps
        assert len(surface_mask.shape) == 3
        assert [*primitive_points.shape[:-1]] == [*surface_mask.shape
                                                  ], (primitive_points.shape,
                                                      surface_mask.shape)

        # B x (Ps * N)
        surface_mask_BxPsN = surface_mask.view(B, -1)

        loss_primitives2target = loss_primitives2target * surface_mask_BxPsN
        if is_normal_loss:
            normal_loss_primitives2target = normal_loss_primitives2target * surface_mask_BxPsN

    loss_primitives2target = loss_primitives2target.mean()

    if is_normal_loss:
        normal_loss_primitives2target = normal_loss_primitives2target.mean()
        normal_loss = (normal_loss_primitives2target +
                       normal_loss_target2primitives) / 2.

    return (loss_primitives2target + loss_target2primitives) / 2., normal_loss


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
