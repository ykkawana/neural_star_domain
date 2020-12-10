import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.nsd import models, training, generation
from im2mesh.nsd import data as nsd_data
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    decoder = models.decoder_dict[decoder](dim=dim,
                                           z_dim=z_dim,
                                           c_dim=c_dim,
                                           **decoder_kwargs)

    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim, **encoder_latent_kwargs)
    else:
        encoder_latent = None

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](c_dim=c_dim, **encoder_kwargs)
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.NeuralStarDomainNetwork(decoder,
                                        encoder,
                                        encoder_latent,
                                        p0_z,
                                        device=device)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(model,
                               optimizer,
                               device=device,
                               input_type=input_type,
                               vis_dir=vis_dir,
                               threshold=threshold,
                               eval_sample=cfg['training']['eval_sample'],
                               **cfg['trainer'])

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
        pnet_point_scale=cfg['trainer']['pnet_point_scale'],
        is_explicit_mesh=cfg['generation'].get('is_explicit_mesh', False),
        is_skip_surface_mask_generation_time=cfg['generation'].get(
            'is_skip_surface_mask_generation_time', False),
        is_just_measuring_time=cfg['generation'].get('is_just_measuring_time',
                                                     False),
        is_fit_to_gt_loc_scale=cfg['generation'].get('is_fit_to_gt_loc_scale',
                                                     False),
        **cfg['generation'].get('mesh_kwargs', {}),
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(torch.zeros(z_dim, device=device),
                       torch.ones(z_dim, device=device))

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    if cfg.get('sdf_generation', False):
        points_transform = None
    with_transforms = cfg['model']['use_camera']

    fields = {}
    fields['points'] = data.PointsField(
        cfg['data']['points_file'],
        points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    if not cfg.get('sdf_generation', False) and cfg['trainer'].get(
            'is_sdf', False):
        sdf_points_transform = data.SubsampleSDFPoints(
            cfg['data']['points_subsample'])
        fields['sdf_points'] = data.SDFPointsField(
            cfg['data']['sdf_points_file'],
            sdf_points_transform,
            with_transforms=with_transforms)

    pointcloud_transform = data.SubsamplePointcloud(
        cfg['data']['pointcloud_target_n'])
    if cfg.get('sdf_generation', False):
        pointcloud_transform = None

    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'],
        pointcloud_transform,
        with_transforms=True)
    fields['angles'] = nsd_data.SphericalCoordinateField(
        cfg['data']['primitive_points_sample_n'],
        mode,
        is_normal_icosahedron=cfg['data'].get('is_normal_icosahedron', False),
        is_normal_uv_sphere=cfg['data'].get('is_normal_uv_sphere', False),
        icosahedron_subdiv=cfg['data'].get('icosahedron_subdiv', 2),
        icosahedron_uv_margin=cfg['data'].get('icosahedron_uv_margin', 1e-5),
        icosahedron_uv_margin_phi=cfg['data'].get('icosahedron_uv_margin_phi',
                                                  1e-5),
        uv_sphere_length=cfg['data'].get('uv_sphere_length', 20),
        normal_mesh_no_invert=cfg['data'].get('normal_mesh_no_invert', False))
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
