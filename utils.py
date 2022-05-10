import torch
import base64
from kornia import create_meshgrid
import torchvision.transforms as T

# global parameter to avoid re-computation
meshgrid_cache = {}


def meshgrid_abs_torch(batch, height, width, device, permute):
    """
    Create a 2D meshgrid with absolute homogeneous coordinates
    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
    """
    global meshgrid_cache
    # avoid cache size being too large
    if len(meshgrid_cache) > 20:
        meshgrid_cache = {}
    key = (batch, height, width, device, permute)
    try:
        res = meshgrid_cache[key]
    except KeyError:
        grid = create_meshgrid(height, width, device=device, normalized_coordinates=False)[0]
        xs, ys = grid.unbind(-1)
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], axis=0)                        # (3, H, W)
        res = coords[None, ...].repeat(batch, 1, 1, 1).to(device=device)    # (B, 3, H, W)
        if permute:
            res = res.permute(0, 2, 3, 1)                                   # (B, H, W, 3)
        meshgrid_cache[key] = res
    return res


def save_image(img, path):
    img = img.permute(2, 0, 1)
    pilImg = T.ToPILImage()(img)
    pilImg.save(path)


def get_base64_encoded_image(image_path):
    """Utils for the HTML viewer"""
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode('utf-8')