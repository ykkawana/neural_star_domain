import os
import glob
import random
import torch
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw
from im2mesh.nsd.data import data_utils

class SphericalCoordinateField(Field):
    ''' Angle field class.

    It provides the class used for spherical coordinate data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self,
                 primitive_points_sample_n,
                 mode,
                 is_normal_icosahedron=False,
                 is_normal_uv_sphere=False,
                 icosahedron_subdiv=2,
                 icosahedron_uv_margin=1e-5,
                 icosahedron_uv_margin_phi=1e-5,
                 uv_sphere_length=20,
                 *args,
                 **kwargs):
        self.primitive_points_sample_n = primitive_points_sample_n
        self.mode = mode
        self.is_normal_icosahedron = is_normal_icosahedron
        self.is_normal_uv_sphere = is_normal_uv_sphere
        self.icosahedron_uv_margin = icosahedron_uv_margin
        self.icosahedron_uv_margin_phi = icosahedron_uv_margin_phi

        if self.is_normal_icosahedron:
            icosamesh = trimesh.creation.icosphere(
                subdivisions=icosahedron_subdiv)

            icosamesh.invert()
            uv = trimesh.util.vector_to_spherical(icosamesh.vertices)

            uv[:, 1] = uv[:, 1] - np.pi / 2

            uv_thetas = torch.tensor(uv).float()
            uv_th = uv_thetas[:, 0].clamp(min=(-np.pi + icosahedron_uv_margin),
                                          max=(np.pi - icosahedron_uv_margin))
            uv_ph = uv_thetas[:, 1].clamp(
                min=(-np.pi / 2 + icosahedron_uv_margin),
                max=(np.pi / 2 - icosahedron_uv_margin))
            self.angles_for_nomal = torch.stack([uv_th, uv_ph], axis=-1)
            self.face_for_normal = torch.from_numpy(icosamesh.faces)

        elif self.is_normal_uv_sphere:
            thetas = data_utils.sample_spherical_angles(
                batch=1,
                sample_num=uv_sphere_length,
                sampling='grid',
                device='cpu',
                dim=3,
                sgn_convertible=True,
                phi_margin=icosahedron_uv_margin_phi,
                theta_margin=icosahedron_uv_margin)
            mesh = trimesh.creation.uv_sphere(
                theta=np.linspace(0, np.pi, uv_sphere_length),
                phi=np.linspace(-np.pi, np.pi, uv_sphere_length))
            mesh.invert()
            self.angles_for_nomal = thetas[0]
            self.face_for_normal = torch.from_numpy(mesh.faces)

    def load(self, model_path, idx, category):
        ''' Sample spherical coordinate.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        angles = data_utils.sample_spherical_angles(
            batch=1,
            sample_num=self.primitive_points_sample_n,
            sampling='grid' if self.mode in ['test', 'val'] else 'uniform',
            device='cpu',
            dim=3,  #sgn_convertible=True, phi_margin=1e-5, theta_margin=1e-5)
            sgn_convertible=True,
            phi_margin=self.icosahedron_uv_margin_phi,
            theta_margin=self.icosahedron_uv_margin).squeeze(0)

        data = {None: angles}
        if self.is_normal_icosahedron or self.is_normal_uv_sphere:
            data.update({
                'normal_angles': self.angles_for_nomal.clone(),
                'normal_face': self.face_for_normal.clone()
            })
        return data

    def check_complete(self, _):
        ''' Check if field is complete.

        Returns: True
        '''
        return True
