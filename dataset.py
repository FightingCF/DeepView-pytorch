from pathlib2 import Path
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from spaces_dataset.code.utils import ReadScene
from PIL import Image


# TODO: is this necessary or not
# TODO: change the src_img dimension from 4 to 5, or there are so many dimensions that equals to 4
STANDARD_LAYOUTS = {   
    'random_4_9': ([0, 3, 9, 12], [6])
}

# TODO: can we just put all data in cuda memory from the very beginning?
# in fact, putting all data in cuda and slice them into batches saves more time?
# all data are in cpu from the very beginning, so there's no need to use "device"
class SpacesDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 layout='random_4_9',
                 n_planes=10,
                 downscale_factor=0.25,
                 near_depth=1,
                 far_depth=100,
                 tiny=False,
                 crop=True,
                 type='train'):
        super().__init__()
        '''
        :param dataset_path: path to the dataset
        :param n_planes: number of planes of MPI
        :param downscale_factor: downscale factor for the images, the MPI resolution
        :param tiny: whether to use a tiny dataset
        :param type: type of the dataset, train or val
        '''
        self.dataset_path = dataset_path
        
        # layout process
        self.layout = layout
        if isinstance(layout, str):
            idices_tuple = STANDARD_LAYOUTS[layout]
        elif isinstance(layout, tuple):
            idices_tuple = layout
        else:
            raise ValueError('SpacesDataset(): layout must be a string or 2-tuple !')
        self.indices_in, self.indices_tgt = idices_tuple
        self.n_in, self.n_tgt = len(self.indices_in), len(self.indices_tgt)
        assert self.n_in > 0 and self.n_tgt > 0, "SpacesDataset(): n_in and n_tgt must be > 0"

        self.n_planes = n_planes
        self.downscale_factor = downscale_factor
        self.near_depth = near_depth
        self.far_depth = far_depth
        self.tiny = tiny
        self.crop = crop
        self.type = type

        self.image_orig_width = 800
        self.image_orig_height = 480

        # Load metadata
        # TODO: do not use 2k resolution now
        scenes_root_folder = Path(dataset_path) / 'data' / '800'
        scenes = [ReadScene(scenes_root_folder / p.name) for p in scenes_root_folder.iterdir()]
        if self.tiny:
            scenes = scenes[:2] if type == 'val' else scenes[10:20]
        else:
            scenes = scenes[:10] if type == 'val' else scenes[10:]
        self.scenes = scenes

        # Create the table with scene id and camera id
        # i = scene index, j = camera id in scene
        self.scene_cam_table = []
        n_scenes = len(self.scenes)
        for i in range(n_scenes):
            for j in range(len(self.scenes[i])):
                self.scene_cam_table.append((i, j))
        # create psv planes
        self.psv_planes = torch.tensor(inv_depths(self.near_depth, self.far_depth, self.n_planes), dtype=torch.float32)


    def __len__(self):
        return len(self.scene_cam_table)


    def __getitem__(self, idx):
        idx_scene, idx_cam = self.scene_cam_table[idx]
        if self.type == 'train' and self.layout == 'random_4_9':
            indices_choices = np.random.choice(len(self.scenes[0][0]), len(self.indices_in) + 1, replace=False)
            
            # indices_choices = np.array([1, 2, 3, 4, 5])
            self.indices_in = indices_choices[:4]
            self.indices_tgt = indices_choices[4:]
        
        indices_in_full = [(idx_scene, idx_cam, image_id) for image_id in self.indices_in]
        indices_tgt_full = [(idx_scene, idx_cam, image_id) for image_id in self.indices_tgt]
        images_info_in = [self.load_image(idx) for idx in indices_in_full]
        images_info_tgt = [self.load_image(idx) for idx in indices_tgt_full]

        res = {
            'images_in': torch.stack([info['image'] for info in images_info_in]).float(),       # (n_in, h, w, c)
            'intrins_in': torch.stack([info['intrin'] for info in images_info_in]).float(),     # (4, 3, 3)
            'w2cs_in': torch.stack([info['w2c'] for info in images_info_in]).float(),           # (4, 4, 4)
            
            'images_tgt': torch.stack([info['image'] for info in images_info_tgt]).float(),     # (1, h, w, 3)
            'intrins_tgt': torch.stack([info['intrin'] for info in images_info_tgt]).float(),   # (1, 3, 3)
            'w2cs_tgt': torch.stack([info['w2c'] for info in images_info_tgt]).float(),         # (1, 4, 4)
            'c2ws_tgt': torch.stack([torch.inverse(info['w2c']) for info in images_info_tgt]).float(), # (1, 4, 4)

            'images_in_orig': torch.stack([info['image'] for info in images_info_in]).float(),       # (n_in, h, w, c)
            'intrins_in_orig': torch.stack([info['intrin'] for info in images_info_in]).float(),     # (4, 3, 3)
            
            'mpi_planes': self.psv_planes,                  # (n_planes,)
        }

        if self.crop:
            # TODO: why tgt_z is shape[0] // 2
            tgt_cam_z = self.psv_planes[self.psv_planes.shape[0] // 2]
            
            # TODO: why add random value to coordinates
            # if self.type == 'train':
            #     tgt_pixel_x = (self.image_orig_width / 2) + \
            #         random.uniform(-0.25, 0.25) * (self.image_orig_width - self.image_orig_width * self.downscale_factor)
            #     tgt_pixel_y = (self.image_orig_height / 2) + \
            #         random.uniform(-0.25, 0.25) * (self.image_orig_height - self.image_orig_height * self.downscale_factor)
            # else:
            #     tgt_pixel_x = (self.image_orig_width / 2)
            #     tgt_pixel_y = (self.image_orig_height / 2)

            tgt_pixel_x = (self.image_orig_width / 2)
            tgt_pixel_y = (self.image_orig_height / 2)

            tgt_pos = (tgt_pixel_x, tgt_pixel_y, tgt_cam_z)

            # elements in res are all cpu tensors
            res = crop_scale_things(res, tgt_pos, self.downscale_factor, self.image_orig_width, self.image_orig_height)

        return res


    def load_image(self, idx):
        '''
        :param idx: (scene_id, camera_id, image_id)
        '''
        data = self.scenes[idx[0]][idx[1]][idx[2]]
        intrin = torch.tensor(data.camera.intrinsics, dtype=torch.float32)
        w2c = torch.tensor(data.camera.c_f_w, dtype=torch.float32)
        image_path = data.image_path
        image = Image.open(image_path).convert('RGB')
        # TODO: can we replace orig_w/h with a simple downscale_factor?
        image, intrin = resize_totensor_intrinsics(image, intrin, self.image_orig_width, self.image_orig_height)
        return {
            'image': image,
            'intrin': intrin,
            'w2c': w2c
        }


# utils for dataset are written below
def inv_depths(near_depth, end_depth, num_depths):
    """
    :param near_depth: the near plane depth
    :param end_depth: the far plane depth
    :param num_depths: the number of planes
    # TODO: why it's useful for back to front compositing
    :return:
        np array (num_depths, )
        the furthest first plane depths, useful for back to front compositing
    """
    return 1.0 / np.linspace(1. / near_depth, 1. / end_depth, num_depths)[::-1]


def scale_intrinsics(intrin, sy, sx):
    '''
    :param intrin: tensor (3, 3)
    :param sy: scale factor for y
    :param sx: scale factor for x
    :return:
        intrin: scaled intrinsics
    '''
    intrin[0, 0] *= sx
    intrin[1, 1] *= sy
    intrin[0, 2] *= sx
    intrin[1, 2] *= sy
    return intrin


def resize_totensor_intrinsics(img, intrin, img_tgt_w, img_tgt_h):
    '''
    :param img: PIL image format
    :param intrin: intrinsics
    :param img_tgt_w: target image width
    :param img_tgt_h: target image height
    :return:
        img: tensor (3, H, W) in range [0, 1]
        intrin: tensor (3, 3)
    '''
    intrin_s = scale_intrinsics(intrin, img_tgt_h / img.height, img_tgt_w / img.width)
    img_s = img.resize((img_tgt_w, img_tgt_h))
    img_t = torch.tensor(np.array(img_s) / 255, dtype=torch.float32)
    return img_t, intrin_s


def pixel_coor_to_world_coor(pixel_x, pixel_y, z, intrin, c2w):
    '''
    :param pixel_x, pixel_y, z: scalar
    :param intrin: (1, 3, 3)
    :param c2w: (1, 4, 4)
    :return:
        world_coor: tensor (4, ) 
    '''
    intrin = intrin[0]
    c2w = c2w[0]
    fx, fy, cx, cy = intrin[0, 0], intrin[1, 1], intrin[0, 2], intrin[1, 2]
    ref_cam_x = z * (pixel_x - cx) / fx
    ref_cam_y = z * (pixel_y - cy) / fy
    ref_cam_center = torch.tensor([ref_cam_x, ref_cam_y, z, 1.0])
    return torch.matmul(c2w, ref_cam_center)


def clamped_world_to_pixel(coor_world, w2c, intrin, crop_x, crop_y, img_w, img_h):
    '''
    :param coor_world: tensor (4, )
    :param w2c: (n_cams, 4, 4)
    :param intrin: (n_cams, 3, 3)
    :param crop_x, crop_y: crop region
    :param img_w, img_h: image size
    :return:
        numpy (n_cams, 2) coordinates 
    '''
    cam_center_in = torch.matmul(w2c, coor_world)
    pixel_center_in = torch.matmul(intrin, cam_center_in[:, :3].unsqueeze(-1)).squeeze(-1)
    pixel_xy_in = torch.div(pixel_center_in[:, :2], pixel_center_in[:, 2].unsqueeze(-1))
    # TODO: why in this range? this is a bug in original code
    pixel_x_in = torch.clamp(pixel_xy_in[:, 0], min=crop_x / 2, max=img_w - crop_x / 2)
    pixel_y_in = torch.clamp(pixel_xy_in[:, 1], min=crop_y / 2, max=img_h - crop_y / 2)
    pixel_xy_in = torch.floor(torch.cat([pixel_x_in.unsqueeze(-1), pixel_y_in.unsqueeze(-1)], dim=1)) # (cams, 2)
    return pixel_xy_in


def crop_intrinsics(intrin, im_w, im_h, crop_cx, crop_cy):
    '''
    :param intrin: [fx, fy, cx, cy]
    :param im_w, im_h: image size
    :param crop_x, crop_y: pos of crop window center
    :return:
        intrin: cropped intrinsics
    '''
    fx, fy, cx, cy = intrin
    cx_new = cx + float(im_w - 1) / 2 - crop_cx
    cy_new = cy + float(im_h - 1) / 2 - crop_cy
    return torch.tensor([
        [fx, 0., cx_new],
        [0., fy, cy_new],
        [0., 0., 1.]
    ], dtype=torch.float32)


def crop_image(img, center_x, center_y, crop_w, crop_h):
    h_min = int(center_y - crop_h / 2)
    h_max = int(center_y + crop_h / 2)
    w_min = int(center_x - crop_w / 2)
    w_max = int(center_x + crop_w / 2)
    return img[h_min:h_max, w_min:w_max]


def crop_scale_things(res, tgt_pos, downscale_factor, image_orig_width, image_orig_height):
    # TODO: why everything changes with the re-center?
    # after deleting all these todos, sort them in CodeTools
    tgt_pixel_x, tgt_pixel_y, tgt_cam_z = tgt_pos
    im_w, im_h = downscale_factor * image_orig_width, downscale_factor * image_orig_height
    tgt_cam_center_world = pixel_coor_to_world_coor(
        tgt_pixel_x, tgt_pixel_y, tgt_cam_z, res['intrins_tgt'], res['c2ws_tgt']
    )
    res['pixel_xy_orig'] = torch.tensor([tgt_pixel_x, tgt_pixel_y], dtype=torch.float32)

    # calculate center pixel coordinates in other cameras
    pixel_xy_in = clamped_world_to_pixel(
        tgt_cam_center_world, res['w2cs_in'], res['intrins_in'], im_w, im_h, image_orig_width, image_orig_height
    )
    res['pixel_xy_in'] = pixel_xy_in

    # adjust intrinsics
    intrins_in_new = []
    for idx, intrin in enumerate(res['intrins_in']):
        intrin_orig = [intrin[0][0], intrin[1][1], intrin[0][2], intrin[1][2]]
        intrin_new = crop_intrinsics(intrin_orig, im_w, im_h, pixel_xy_in[idx, 0], pixel_xy_in[idx, 1])
        intrins_in_new.append(intrin_new.unsqueeze(0))
    res['intrins_in'] = torch.cat(intrins_in_new, dim=0)

    # crop images
    images_in_new = []
    for idx, img in enumerate(res['images_in']):
        img_cropped = crop_image(img, pixel_xy_in[idx, 0], pixel_xy_in[idx, 1], im_w, im_h)
        images_in_new.append(img_cropped.unsqueeze(0))
    res['images_in'] = torch.cat(images_in_new, dim=0)

    # adjust tgt intrinsics
    intrins_tgt_orig = [res['intrins_tgt'][0][0, 0], res['intrins_tgt'][0][1, 1], res['intrins_tgt'][0][0, 2], res['intrins_tgt'][0][1, 2]]
    res['intrins_tgt'] = crop_intrinsics(intrins_tgt_orig, im_w, im_h, tgt_pixel_x, tgt_pixel_y).unsqueeze(0)

    # adjust tgt images
    res['images_tgt'] = crop_image(
        res['images_tgt'][0], tgt_pixel_x, tgt_pixel_y, im_w, im_h
    ).unsqueeze(0)

    return res
