import torch
import warnings
import math
import numpy as np

EPS = 1e-7
def generate_grid_samples(grid_size,
                          batch=1,
                          sampling='grid',
                          sample_num=100,
                          device='cpu',
                          dim=2):
    """Generate mesh grid.
    Args:
        grid_size (int, list, dict):
            if type is int, sample between [-grid_size, grid_size] range for all
            dims.
            if type is list, sample between grid_size for all dims. 
            if type is dict, For each dim, sample between each element of `range` in grid_size 
            with number of samples specified by `sample_num` in grid_size.
        batch (int): batch size. Sample size will be batch size * grid samples
        sampling (str): sample gridwise if it's `grid`, sample by uniform dist. if it's `uniform`.
        sample_num (int): number of samples. If grid_type is dict, this argument will be ignored.
        device: if set, data will be initialized on that device. otherwise will be stored on `cpu`.
        dim: dimension of meshgrid. Defaults to 2.
    Returns:
        coord (B, sample size, dim)
    """
    if isinstance(grid_size, list):
        sampling_start, sampling_end = grid_size
    elif isinstance(grid_size, int):
        sampling_start, sampling_end = -grid_size, grid_size
    else:
        dim = len(grid_size['range'])
        assert dim == len(grid_size['sample_num'])
        warnings.warn('sample_num and dim will be ignored')

    def contiguous_batch(s):
        return s.contiguous().view(1, -1).repeat(batch, 1)

    ranges = []

    if sampling == 'grid':
        for idx in range(dim):
            if isinstance(grid_size, dict):
                sampling_start, sampling_end = grid_size['range'][idx]
                sample_num = grid_size['sample_num'][idx]
            sampling_points = torch.linspace(sampling_start,
                                             sampling_end,
                                             sample_num,
                                             device=device)
            ranges.append(sampling_points)

        coord_list = [contiguous_batch(s) for s in torch.meshgrid(ranges)]
        return torch.stack(coord_list, axis=-1)
    elif sampling == 'uniform':
        if isinstance(grid_size, dict):
            total_sample_num = np.prod(grid_size['sample_num'])
        else:
            total_sample_num = sample_num**dim
        sampling_points = torch.empty([batch, total_sample_num, dim],
                                      device=device).uniform_(0, 1)

        def normalize(x, bounds):
            return bounds[0] + (x - 0.) * (bounds[1] - bounds[0]) / (1. - 0.)

        if isinstance(grid_size, int):
            return sampling_points * grid_size
        elif isinstance(grid_size, list):
            return normalize(sampling_points, grid_size)
        elif isinstance(grid_size, dict):
            ranged_sampling_points_list = []
            for idx in range(dim):
                ranged_sampling_points_list.append(
                    normalize(sampling_points[..., idx],
                              grid_size['range'][idx]))
            return torch.stack(ranged_sampling_points_list, axis=-1)
    else:
        raise NotImplementedError('no such sampling mode')


def sample_spherical_angles(batch=1,
                            sampling='grid',
                            sample_num=100,
                            device='cpu',
                            phi_margin=EPS,
                            theta_margin=0,
                            sgn_convertible=False,
                            spherical_coord=False,
                            dim=2):
    assert dim in [2, 3]
    if not sgn_convertible:
        phi_margin = 0
    grid_range = {'range': [[-math.pi, math.pi]], 'sample_num': [sample_num]}
    if dim == 2:
        grid_range = {
            'range': [[-math.pi, math.pi]],
            'sample_num': [sample_num]
        }
        #grid_range = {'range': [[0., 2 * math.pi]], 'sample_num': [sample_num]}
    else:
        grid_range = {
            'range': [[-math.pi + theta_margin, math.pi - theta_margin],
                      [-math.pi / 2 + phi_margin, math.pi / 2 - phi_margin]],
            'sample_num': [sample_num, sample_num]
        }
    if spherical_coord:
        assert dim == 3
        grid_range = {
            'range': [[0, 2 * math.pi], [0, math.pi]],
            'sample_num': [sample_num, sample_num]
        }
    angles = generate_grid_samples(grid_range,
                                   sampling=sampling,
                                   device=device,
                                   batch=batch,
                                   dim=dim - 1)
    return angles
