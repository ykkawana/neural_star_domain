import torch
from im2mesh.nsd.models import base_shape_decoder


class SphereDecoder(base_shape_decoder.BaseShapeDecoder):
    def __init__(self, *args, **kwargs):
        """Intitialize SuperShapeSampler.

        Args:
            max_m: Max number of vertices in a primitive. If quadrics is True,
                then max_m must be larger than 4.
            rational: Use rational (stable gradient) version of supershapes
        """
        super().__init__(*args, **kwargs)

    def get_r(self, thetas, *args, **kwargs):
        """Return radius.

        B, _, P, D = thetas.shape
        Args:
            thetas: (B, 1 or n_primitives, P, D)
        Returns:
            radius: (B, n_primitives, P, D)
        """
        B, _, P, _ = thetas.shape

        #r = (B, n_primitives, max_m, P, dim-1), thetas = (B, 1 or N, 1, P, dim-1)
        r = torch.ones([B, self.n_primitives, P, self.dim - 1],
                       device=thetas.device,
                       dtype=thetas.dtype)
        return r
