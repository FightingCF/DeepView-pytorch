from statistics import mode
import numpy as np
import torch
import torch.nn.functional as F
from utils import meshgrid_abs_torch


def divide_safe_torch(fir, sec):
    """
    Operate fir / sec, but avoid dividing by zero
    """
    eps = 1e-8
    sec = sec.float()
    sec += eps * sec.eq(torch.tensor(0, device=sec.device, dtype=torch.float32))
    return torch.div(fir.float(), sec)


# TODO: the computation procedure is ambiguous
def inv_homography_torch(intrin_src, intrin_tgt, rot, t, plane_normal, minus_depths):
    """
    Compute inverse homography
    Args:
        intrin_src: (n_planes, B, 3, 3)
        intrin_tgt: (n_planes, B, 3, 3)
        rot: (n_planes, B, 3, 3)
        t: (n_planes, B, 3, 1)
        plane_normal: (n_planes, B, 1, 3)
        minus_depths: (n_planes, B, 1, 1)
    Returns:
        inv_homography: (n_planes, B, 3, 3)
    """
    rot_t = rot.transpose(2, 3)     # clockwise <-> counterclockwise
    intrin_tgt_inv = torch.inverse(intrin_tgt)

    sec = minus_depths - (plane_normal @ rot_t) @ t
    fir = ((rot_t @ t) @ plane_normal) @ rot_t
    inv_hom = (intrin_src @ (rot_t + divide_safe_torch(fir, sec))) @ intrin_tgt_inv
    return inv_hom


def homowarp_coords_torch(coords, homography):
    """
    Transform input coords according to homography
    Args:
        coords: (n_planes, B, H, W, 3) pixel coordinates (u, v, 1)
        homography: (n_planes, B, 3, 3)
    Returns:
        transformed coords: (n_planes, B, H, W, 3)
    """
    n_planes, B, H, W, _ = coords.shape
    coords = coords.reshape(n_planes, B, -1, 3) # (n_planes, B, H*W, 3)
    transformed_coords = coords @ homography.transpose(2, 3)
    transformed_coords = transformed_coords.reshape(n_planes, B, H, W, 3)
    return transformed_coords


def normalized_pixel_coords_torch(coords):
    """
    Convert homogeneous pixel coords to regular coords
    Args:
        coords: (..., 3)
    Return:
        coords_uv_norm: (..., 2)
    """
    uv = coords[..., :-1]
    w = coords[..., -1:]
    return divide_safe_torch(uv, w)


def bilinear_sampler_torch(rgba_layers, coords):
    """
    Bilinear sampler function
    # TODO: what is the shape of rgba_layers?
    Args:
        rgba_layers: (n_src_imgs, n_planes, H, W, 4)
        coords: (n_src_imgs, H_, W_, 2)
    Returns:
        sampled_rgba: (n_src_imgs, H_, W_, 3)
    """
    n_planes, n_src_imgs, H, W, C = rgba_layers.shape
    _, _, H_, W_, _ = coords.shape
    rgba_layers = rgba_layers.reshape(n_src_imgs * n_planes, H, W, C)
    coords = coords.reshape(-1, H_,  W_, 2)
    
    rgba_layers = rgba_layers.permute(0, 3, 1, 2)
    coords2 = torch.tensor([-1, -1], dtype=torch.float32, device=rgba_layers.device) + 2 * coords

    rgba_layers_sampled = F.grid_sample(rgba_layers, coords2, align_corners=True)
    rgba_layers_sampled = rgba_layers_sampled.reshape(n_planes, n_src_imgs, *rgba_layers_sampled.shape[-3:])

    return rgba_layers_sampled


def project_forward_homography_torch(rgba_layers, intrin_tgt, intrin_src, pose_tgt, depths):
    """
    Use homography for forward warping
    Args:
        # TODO: Change all B that should be n_src_imgs
        rgba_layers: (n_planes, n_src_imgs, H, W, 4)
        intrin_tgt: (n_src_imgs, 3, 3)
        intrin_src: (n_src_imgs, 3, 3)
        pose_tgt: (n_src_imgs, 4, 4)
        depths: (n_planes, n_src_imgs)
    Returns:
        proj_rgba_layers: (n_planes, n_src_imgs, 4, H, W)
    """
    n_planes, n_src_imgs, H, W, _ = rgba_layers.shape
    pixel_coords_tgt = meshgrid_abs_torch(n_src_imgs, H, W, device=rgba_layers.device, permute=True)
    rot = pose_tgt[:, :3, :3]
    t = pose_tgt[:, :3, 3]
    plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=rgba_layers.device).reshape(1, 1, 1, 3)
    plane_normal = plane_normal.expand(n_planes, n_src_imgs, -1, -1)

    minus_depths = -depths.reshape(n_planes, n_src_imgs, 1, 1)

    # Add the first dimension and set it to n_planes
    intrin_src = intrin_src[None, ...].expand(n_planes, -1, -1, -1)
    intrin_tgt = intrin_tgt[None, ...].expand(n_planes, -1, -1, -1)
    t = t[None, :, :, None].expand(n_planes, -1, -1, -1)
    rot = rot[None, ...].expand(n_planes, -1, -1, -1)
    pixel_coords_tgt = pixel_coords_tgt[None, ...].expand(n_planes, -1, -1, -1, -1)

    homo_tgt2src_planes = inv_homography_torch(intrin_src, intrin_tgt, rot, t, plane_normal, minus_depths)
    pixel_coords_tgt2src = homowarp_coords_torch(pixel_coords_tgt, homo_tgt2src_planes)
    pixel_coords_tgt2src = normalized_pixel_coords_torch(pixel_coords_tgt2src)

    # a more accurate way to make pixel_coords_tgt2src in [0, 1]
    # w_max, w_min = pixel_coords_tgt2src[..., 0].max(), pixel_coords_tgt2src[..., 0].min()
    # h_max, h_min = pixel_coords_tgt2src[..., 1].max(), pixel_coords_tgt2src[..., 1].min()
    # pixel_coords_tgt2src[..., 0] = (pixel_coords_tgt2src[..., 0] - w_min) / (w_max - w_min)
    # pixel_coords_tgt2src[..., 1] = (pixel_coords_tgt2src[..., 1] - h_min) / (h_max - h_min)
    pixel_coords_tgt2src = pixel_coords_tgt2src / torch.tensor([W - 1, H - 1], device=rgba_layers.device)

    rgba_layers_src2tgt = bilinear_sampler_torch(rgba_layers, pixel_coords_tgt2src)

    return rgba_layers_src2tgt


def over_composite(rgbas):
    """
    Combines a list of RGBA images using the over operation.

    Combines RGBA images from back to front with the over operation.
    The alpha image of the first image is ignored and assumed to be 1.0.

    Args:
      rgbas: A list of [B, H, W, 4] RGBA images, combined from back to front.
    Returns:
      Composited RGB image.
    """
    for i in range(len(rgbas)):
        rgb = rgbas[i][:, :, :, 0:3]
        alpha = rgbas[i][:, :, :, 3:]
        if i == 0:
            output = rgb
        else:
            rgb_by_alpha = rgb * alpha
            output = rgb_by_alpha + output * (1.0 - alpha)
    return output


def mpi_render_view_torch(rgba_layers, pose_tgt, planes, intrin_tgt, intrin_src):
    """
    Render a target view from an MPI representation
    
    Args:
        rgba_layers: input MPI (n_src_imgs, H, W, n_planes, 4)
        pose_tgt: target pose to render from (n_src_imgs, 4, 4)
        planes: list of depths for each plane (n_planes, )
        intrin_tgt: target camera intrinsics (n_src_imgs, 3, 3)
        intrin_src: source camera intrinsics (n_src_imgs, 3, 3)
    
    Returns:
        rendered target view (n_src_imgs, H, W, 3)
    """
    n_src_imgs, _, _ = pose_tgt.shape
    depths = planes[:, None]                # (n_planes, 1)
    depths = depths.repeat(1, n_src_imgs)   # (n_planes, n_src_imgs)

    rgba_layers = rgba_layers.permute(3, 0, 1, 2, 4)        # (n_planes, n_src_imgs, H, W, 4)
    proj_images = project_forward_homography_torch(rgba_layers, intrin_tgt, intrin_src, pose_tgt, depths)
    proj_images = proj_images.permute(0, 1, 3, 4, 2)
    proj_images_list = []

    for i in range(len(planes)):
        proj_images_list.append(proj_images[i])
    output_image = over_composite(proj_images_list)
    return output_image
