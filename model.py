from cv2 import accumulate
from sympy import comp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import meshgrid_abs_torch
import matplotlib.pyplot as plt
from render import project_forward_homography_torch

# global parameter to avoid re-computation
meshgrid_cache = {}


def conv_layer(in_channels, out_channels, kernel_size, stride, transpose=False, bias=False, padding=None, dilation=1):
    # refer to fast.ai ConvLayer
    if padding is None:
        padding = (kernel_size - 1) // 2 if not transpose else 0
    if transpose:
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation)
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation)
    return nn.Sequential(
        conv, 
        nn.ELU(),
        nn.BatchNorm2d(out_channels)
    )


class ModelBlock(nn.Module):
    # TODO: in_channels seems not to be the meaning I thought
    def __init__(self, 
                 in_channels=8,
                 iteration_num=0,
                 is_final_iterarion=False):
        super().__init__()

        self.iteration_num = iteration_num
        self.is_final_iterarion = is_final_iterarion
        if self.iteration_num == 1:
            # (B, C, H, W) -> (B, 4C, H // 2, W // 2)
            # TODO: this SpaceToDepth is just a downsampled block, why bother so complicated
            self.pre_cnn = nn.Sequential(
                conv_layer(in_channels, in_channels * 2, kernel_size=1, stride=1),
                SpaceToDepth(2)
            )
            self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear')
        elif self.iteration_num > 1:
            self.upsampler = nn.Identity()  # does not change anything
        
        cnn1_in_channels = 14 if self.iteration_num > 0 else 3
        # TODO: why up and down in cnn1, and how about a larger kernel size?
        self.cnn1 = nn.Sequential(
            conv_layer(cnn1_in_channels, 2 * in_channels, kernel_size=3, stride=1),
            SpaceToDepth(2),
            conv_layer(8 * in_channels, 4 * in_channels, kernel_size=3, stride=1),
            conv_layer(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1),
            conv_layer(8 * in_channels, 8 * in_channels, kernel_size=3, stride=1),
            conv_layer(8 * in_channels, 3 * in_channels, kernel_size=3, stride=1),
            conv_layer(3 * in_channels, 12 * in_channels, kernel_size=3, stride=1)
        ) if self.iteration_num < 2 else nn.Sequential(
            conv_layer(cnn1_in_channels, 4 * in_channels, kernel_size=3, stride=1),
            conv_layer(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1),
            conv_layer(8 * in_channels, 8 * in_channels, kernel_size=3, stride=1),
            conv_layer(8 * in_channels, 3 * in_channels, kernel_size=3, stride=1),
            conv_layer(3 * in_channels, 12 * in_channels, kernel_size=3, stride=1)
        )

        self.cnn2 = nn.Sequential(
            conv_layer(24 * in_channels, 12 * in_channels, kernel_size=1, stride=1),
            conv_layer(12 * in_channels, 8 * in_channels, kernel_size=1, stride=1)
        )

        self.cnn3 = nn.Sequential(
            conv_layer(16 * in_channels, 16 * in_channels, kernel_size=1, stride=1),
            conv_layer(16 * in_channels, 12 * in_channels, kernel_size=1, stride=1)
        )

        cnn4_param = 8 if self.iteration_num < 2 else 2
        cnn4_extra_in = 0 if iteration_num == 0 else (16 * 4 if iteration_num == 1 else 8)

        cnn4_out_dim = 4 if self.is_final_iterarion else 8
        self.cnn4 = nn.Sequential(
            conv_layer(12 * in_channels + cnn4_extra_in, cnn4_param * in_channels, kernel_size=3, stride=1),
            conv_layer(cnn4_param * in_channels, cnn4_param * in_channels, kernel_size=3, stride=1),
            # (B, r * r * C, H, W) -> (B, C, r * H, r * W)
            nn.PixelShuffle(2),
            nn.Conv2d(cnn4_param * in_channels // 4, cnn4_out_dim, kernel_size=3, stride=1, padding=1)
        ) if self.iteration_num < 2 else nn.Sequential(
            conv_layer(12 * in_channels + cnn4_extra_in, cnn4_param * in_channels, kernel_size=3, stride=1),
            conv_layer(cnn4_param * in_channels, cnn4_param * in_channels, kernel_size=3, stride=1),
            nn.Conv2d(cnn4_param * in_channels, cnn4_out_dim, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, input):
        # TODO: in and src should be the same, correct them in the datasets
        images = input['images_in']
        B, _, H, W, _ = images.shape

        psvs = []
        for i in range(B):
            poses = torch.matmul(input['w2cs_in'][i], input['c2ws_tgt'][i])
            psv = plane_sweep_torch(input['images_in_orig'][i], input['mpi_planes'][i], poses, 
                input['intrins_in_orig'][i], input['intrins_tgt'][i], H, W)
            psvs.append(psv)
        psvs = torch.cat(psvs, dim=0)

        depth_batches, n_views, _, _, _ = psvs.shape

        if self.iteration_num > 0:
            pre_in = input['mpi'].contiguous().view(depth_batches, -1, H, W)
            pre_out = self.pre_cnn(pre_in) if self.iteration_num == 1 else pre_in
            # TODO: why sigmoid
            # TODO: what about the other 60 channels?
            rgba_mpis = rgba_premultiply(torch.sigmoid(self.upsampler(pre_out[:, :4])))

            mpi_gradients = []
            for i in range(B):
                mpi_gradient = calculate_mpi_gradient(
                    input['mpi'][i],
                    input['c2ws_tgt'][i],
                    input['mpi_planes'][i],
                    input['w2cs_in'][i],
                    input['intrins_in'][i],
                    input['images_in'][i],
                    input['intrins_tgt'][i]
                )
                gradient_channels = mpi_gradient.shape[2]
                mpi_gradients.append(mpi_gradient)
            mpi_gradients = torch.cat(mpi_gradients, dim=0)[:, :, :7]
            psvs_gradients = psvs.permute(0, 1, 4, 2, 3)
            mpi_gradients = torch.cat([mpi_gradients, psvs_gradients], dim=2)

            cnn1_in = torch.cat([
                mpi_gradients, rgba_mpis.unsqueeze(1).expand(-1, n_views, -1, -1, -1)
            ], 2).contiguous().view(depth_batches * n_views, gradient_channels + 4, H, W)
        else:
            cnn1_in = psvs.permute(0, 1, 4, 2, 3).reshape(depth_batches * n_views, -1, H, W)
        
        cnn1_out = self.cnn1(cnn1_in)   
        _, channels, h, w = cnn1_out.shape
        maxk0 = cnn1_out.contiguous().view(depth_batches, n_views, channels, h, w).max(dim=1)[0] \
            .unsqueeze(1).expand(-1, n_views, -1, -1, -1).contiguous().view(depth_batches * n_views, channels, h, w)
        
        cnn2_in = torch.cat([cnn1_out, maxk0], dim=1)
        cnn2_out = self.cnn2(cnn2_in)
        _, channels, h, w = cnn2_out.shape
        maxk1 = cnn2_out.contiguous().view(depth_batches, n_views, channels, h, w).max(dim=1)[0] \
            .unsqueeze(1).expand(-1, n_views, -1, -1, -1).contiguous().view(depth_batches * n_views, channels, h, w)
        
        cnn3_in = torch.cat([cnn2_out, maxk1], dim=1)
        cnn3_out = self.cnn3(cnn3_in)
        _, channels, h, w = cnn3_out.shape
        maxk2 = cnn3_out.contiguous().view(depth_batches, n_views, channels, h, w).max(dim=1)[0]

        if self.iteration_num > 0:
            cnn4_in = torch.cat([maxk2, pre_out], dim=1)
        else:
            cnn4_in = maxk2 
        cnn4_out = self.cnn4(cnn4_in)
        _, channels, h, w = cnn4_out.shape
        cnn4_out = cnn4_out.view(B, depth_batches // B, channels, h, w)

        return cnn4_out


class DeepViewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = ModelBlock(iteration_num=0)
        self.model1 = ModelBlock(iteration_num=1)
        self.model2 = ModelBlock(iteration_num=2, is_final_iterarion=True)

    def forward(self, input):
        out0 = self.model0(input)
        
        in1 = {'mpi': out0, **input}
        gradient0 = self.model1(in1)
        out1 = out0 + gradient0

        in2 = {'mpi': out1, **input}
        gradient1 = self.model2(in2)
        out2 = out1[:, :, :4] + gradient1

        return out2

# model utils
class SpaceToDepth(nn.Module):
    """
    This class downsample the input tensor by downscale_factor and 
    then unfold it to a tensor of size (B, C*downscale_factor*downscale_factor, H//downscale_factor, W//downscale_factor)
    """
    def __init__(self, downscale_factor):
        '''
        :param downscale_factor: should be a even number
        '''
        super().__init__()
        self.downscale_factor = downscale_factor
    
    def forward(self, x):
        '''
        :param x: tensor (B, C, H, W)
        '''
        n, c, h, w = x.size()
        unfolded_x = F.unfold(x, kernel_size=self.downscale_factor, stride=self.downscale_factor)
        return unfolded_x.view(n, c * self.downscale_factor * self.downscale_factor,
            h // self.downscale_factor, w // self.downscale_factor)


def pixel2cam_torch(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """
    Convert pixel coordinates to camera coordinates (i.e. 3D points).
    Args:
        depth: depth maps -- [B, H, W]
        pixel_coords: pixel coordinates -- [B, 3, H, W]
        intrinsics: camera intrinsics -- [B, 3, 3]
    Returns:
        cam_coords: points in camera coordinates -- [B, 3 (4 if homogeneous), H, W]
    """
    B, H, W = depth.shape
    depth = depth.reshape(B, 1, -1)
    pixel_coords = pixel_coords.reshape(B, 3, -1)
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth

    if is_homogeneous:
        ones = torch.ones((B, 1, H*W), device=cam_coords.device)
        cam_coords = torch.cat([cam_coords, ones], dim=1)
    cam_coords = cam_coords.reshape(B, -1, H, W)
    return cam_coords


def cam2pixel_torch(cam_coords, proj):
    """
    Convert camera coordinates to pixel coordinates.
    Args:
        cam_coords: points in camera coordinates -- [B, 4, H, W]
        proj: camera intrinsics -- [B, 4, 4]
    Returns:
        pixel_coords: points in pixel coordinates -- [B, H, W, 2]
    """
    B, _, H, W = cam_coords.shape
    cam_coords = torch.reshape(cam_coords, [B, 4, -1])
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)
    xy_u = unnormalized_pixel_coords[:, :2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    pixel_coords = xy_u / (z_u + 1e-10)     # safe division
    pixel_coords = torch.reshape(pixel_coords, [B, 2, H, W])
    return pixel_coords.permute(0, 2, 3, 1)


def resampler_wrapper_torch(images, coords):
    """
    equivalent to tfa.image.resampler
    Args:
        images: [B, H, W, C]
        coords: [B, H, W, 2] source pixel coords 
    """
    return F.grid_sample(
        images.permute([0, 3, 1, 2]),
        torch.tensor([-1, -1], device=images.device) + 2. * coords,
        align_corners=True
    ).permute([0, 2, 3, 1])


# TODO: rename to project_src_to_tgt_planes
def project_inverse_warp_torch(imgs, depth_map, poses, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width):
    """
    Inverse warp a source image to the target image plane based on projection
    Args:
        imgs: source images (n_src_imgs, H, W, 3)
        depth_map: depth map of the target image (n_src_imgs, H, W)
        poses: target to source camera transformation (n_src_imgs, 4, 4)
        src_intrinsics: source camera intrinsics (n_src_imgs, 3, 3)
        tgt_intrinsics: target camera intrinsics (n_src_imgs, 3, 3)
        tgt_height: target image height
        tgt_width: target image width
    Returns:
        imgs_warped: warped images (n_src_imgs, H, W, 3)
    """
    n_src_imgs, H, W, _ = imgs.shape
    pixel_coords = meshgrid_abs_torch(n_src_imgs, tgt_height, tgt_width, imgs.device, False) # (B, 3, H, W)
    cam_coords = pixel2cam_torch(depth_map, pixel_coords, tgt_intrinsics)

    # Construct a 4 x 4 intrinsic matrix
    src_intrinsics4 = torch.zeros(n_src_imgs, 4, 4, device=imgs.device)
    src_intrinsics4[:, :3, :3] = src_intrinsics
    src_intrinsics4[:, 3, 3] = 1

    proj_tgt_cam_to_src_pixel = torch.matmul(src_intrinsics4, poses)
    src_pixel_coords = cam2pixel_torch(cam_coords, proj_tgt_cam_to_src_pixel)
    
    # rewrite to make src_pixel_coords in [0, 1]
    # w_max, w_min = src_pixel_coords[..., 0].max(), src_pixel_coords[..., 0].min()
    # h_max, h_min = src_pixel_coords[..., 1].max(), src_pixel_coords[..., 1].min()
    # src_pixel_coords[..., 0] = (src_pixel_coords[..., 0] - w_min) / (w_max - w_min)
    # src_pixel_coords[..., 1] = (src_pixel_coords[..., 1] - h_min) / (h_max - h_min)
    src_pixel_coords = src_pixel_coords / torch.tensor([W - 1, H - 1], device=imgs.device)

    output_imgs = resampler_wrapper_torch(imgs, src_pixel_coords)
    return output_imgs


def plane_sweep_torch(imgs, depth_planes, poses, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width):
    """
    Construct a plane sweep volume
    
    Args:
        imgs: source images (n_src_imgs, h, w, c)
        depth_planes: a list of depth_values for each plane (n_planes, )
        poses: target to source camera transformation (n_src_imgs, 4, 4)
        src_intrinsics: source camera intrinsics (n_src_imgs, 3, 3)
        tgt_intrinsics: target camera intrinsics (n_src_imgs, 3, 3)
        tgt_height: target image height
        tgt_width: target image width
    Returns:
        volume: a tensor of size (n_planes, n_src_imgs, height, width, c)
    """
    # TODO: batch is not n_src_imgs?
    n_src_imgs = imgs.shape[0]
    plane_sweep_volume = []

    # TODO: this whole warping operation needs to be collected
    for depth in depth_planes:
        curr_depth = torch.zeros([n_src_imgs, tgt_height, tgt_width], dtype=torch.float32, device=imgs.device) + depth
        warped_imgs = project_inverse_warp_torch(imgs, curr_depth, poses, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width)
        plane_sweep_volume.append(warped_imgs)
    plane_sweep_volume = torch.stack(plane_sweep_volume, dim=0)
    return plane_sweep_volume


def show_psv(psvs):
    n_depths, n_images, H, W, C = psvs.shape
    plt.figure(figsize=(19, 6))
    for i_image in range(n_images):
        for i_depth in range(n_depths):
            plt.subplot(n_images, n_depths, i_image * n_depths + i_depth + 1)
            t = psvs[i_depth, i_image]
            plt.imshow((t.detach().cpu().numpy() * 255).astype('uint8'))
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def rgba_premultiply(rgba_layers):
    """
    Premultiply the RGB channels with the alpha channel
    Args:
        rgba_layers: (B, 4, H, W) in [0, 1]
    Returns:
        premultiplied_rgb and alpha (B, 4, H, W) in [0, 1]
    """
    # premultiplied_rgb = rgba_layers[:, :3, :, :] * rgba_layers[:, 3, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
    # a simplified version
    premultiplied_rgb = rgba_layers[:, :3, :, :] * rgba_layers[:, 3:4, :, :]
    return torch.cat([premultiplied_rgb, rgba_layers[:, 3, :, :].unsqueeze(1)], dim=1)


def repeated_over(colors, alpha, n_planes):
    """

    Args:
        colors: (n_planes, n_src_imgs, H, W, 3) in [0, 1]
        alpha: (n_planes, n_src_imgs, H, W) in [0, 1]
        n_planes: integer
    """
    if n_planes < 1:
        return colors[0] * alpha[0][..., None]
    complement_alpha = 1. - alpha
    premultiplied_colors = colors * alpha[..., None]
    return torch.sum(torch.stack([premultiplied_colors[i] * torch.prod(complement_alpha[i + 1: n_planes], 0).unsqueeze(-1).expand(-1, -1, -1, 3)
            for i in range(n_planes)]), dim=0)


# TODO: in fact, mpi_planes should be replaced by mpi_depths
# TODO: review this function and its sub-functions after training
def calculate_mpi_gradient(raw_mpis, c2ws_tgt, mpi_planes, w2cs_src, intrins_src, imgs_src, intrins_tgt):
    """
    Calculate the gradient of MPI
    Args:
        raw_mpis: input MPI (n_planes, C, H, W)
        c2ws_tgt: camera to world coordinates (4, 4)
        mpi_planes: MPI depths (n_planes)
        w2cs_src: world to camera coordinates (n_src_imgs, 4, 4)
        intrins_src: source camera intrinsics (n_src_imgs, 3, 3)
        imgs_src: source images (n_src_imgs, H, W, C)
        intrins_tgt: target camera intrinsics (3, 3)
    Returns:
        grad_mpi: gradient of MPI (n_planes, n_src_imgs, C, H, W)
    """
    n_planes, _, H, W = raw_mpis.shape
    n_src_imgs = w2cs_src.shape[0]
    pose_tgt = w2cs_src @ c2ws_tgt
    rgba_layers = raw_mpis[:, :4][None, ...].expand(n_src_imgs, -1, -1, -1, -1)

    proj_images = project_forward_homography_torch(
        rgba_layers.permute(1, 0, 3, 4, 2),
        intrins_tgt.expand(n_src_imgs, -1, -1),
        intrins_src,
        pose_tgt,
        mpi_planes[:, None].expand(-1, n_src_imgs)
    ).permute(0, 1, 3, 4, 2)    # (n_planes, n_src_imgs, H, W, 4)

    colors = proj_images[:, :, :, :, :3]
    normalized_alphas = proj_images[:, :, :, :, 3]
    complement_alpha = 1. - normalized_alphas

    net_transmittance = torch.stack([torch.prod(complement_alpha[d + 1:], 0) for d in range(n_planes)])
    accumulated_over = torch.stack([repeated_over(colors, normalized_alphas, d-1) for d in range(n_planes)])
    broadcasted_over = repeated_over(colors, normalized_alphas, n_planes)[None, ...].expand(n_planes, -1, -1, -1, -1)
    broadcasted_imgs_src = imgs_src[None, ...].expand(n_planes, -1, -1, -1, -1)

    stacked_input = torch.cat([
        net_transmittance[..., None],
        accumulated_over,
        broadcasted_over,
        broadcasted_imgs_src
    ], dim=-1)

    calculated_depths = torch.zeros([n_planes, H, W], dtype=torch.float32, device=mpi_planes.device) + mpi_planes.contiguous().view(n_planes, 1, 1)
    calculated_depths = calculated_depths.unsqueeze(1).expand(-1, n_src_imgs, -1, -1).contiguous().view(n_planes * n_src_imgs, H, W)
    calculated_pose = pose_tgt[None, ...].expand(n_planes, -1, -1, -1).contiguous().view(n_planes * n_src_imgs, 4, 4)
    calculated_tgt_intrinsics = intrins_tgt.expand(n_planes * n_src_imgs, -1, -1)
    calculated_src_intrinsics = intrins_src[None, ...].expand(n_planes, -1, -1, -1).contiguous().view(n_planes * n_src_imgs, 3, 3)

    return project_inverse_warp_torch(
        stacked_input.contiguous().view(n_planes * n_src_imgs, H, W, 10),
        calculated_depths,
        calculated_pose,
        calculated_src_intrinsics,
        calculated_tgt_intrinsics,
        H, W
    ).contiguous().view(n_planes, n_src_imgs, H, W, 10).permute(0, 1, 4, 2, 3)
