import torch
from dataset import SpacesDataset
from model import DeepViewModel
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from render import mpi_render_view_torch
from vgg_loss import VGGPerceptualLoss1
from pathlib2 import Path
from utils import save_image, get_base64_encoded_image


class Trainer:
    def __init__(self,
                 dset_dir,
                 epochs,
                 batch_size,
                 lr,
                 downscale_factor,
                 n_planes,
                 device):
        self.dset_dir = dset_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.downscale_factor = downscale_factor
        self.n_planes = n_planes
        self.device = device
        
        self.model = DeepViewModel().to(device=device)
        self.vgg_loss = VGGPerceptualLoss1(resize=False, device=device)

        self.dset_train = SpacesDataset(dset_dir, n_planes=self.n_planes, 
            downscale_factor=self.downscale_factor, type='train')
        self.dset_val = SpacesDataset(dset_dir, n_planes=self.n_planes,
            downscale_factor=self.downscale_factor, type='val')
        
        self.loader_train = DataLoader(self.dset_train, batch_size=self.batch_size, shuffle=True)
        self.loader_val = DataLoader(self.dset_val, batch_size=self.batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epoch(self,):
        self.model.train()
        loss_sum = 0
        for input in tqdm(self.loader_train):
            input = tensor_to_device(input, self.device)

            # the 5 lines below should be a template for the forward pass
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss(output, input)
            loss.backward()
            self.optimizer.step()

            # NOTE: use loss.item() except in backward()
            loss_sum += loss.item()
        return loss_sum / len(self.loader_train)

    def render_image(self, output, input):
        """
        Render an image from the output of the model
        """
        # TODO: why sigmoid here out of the network?
        output = torch.sigmoid(output)
        rgba_layers = output.permute(0, 3, 4, 1, 2)
        n_targets = input['images_tgt'].shape[1]

        batch_size = rgba_layers.shape[0]
        images_render = []
        for i in range(batch_size):
            pose = torch.matmul(input['w2cs_tgt'][i], input['c2ws_tgt'][i])
            intrin_tgt = input['intrins_tgt'][i]
            intrin_src = input['intrins_tgt'][i]
            rgba = rgba_layers[i].unsqueeze(0).repeat(n_targets, 1, 1, 1, 1)
            
            # TODO: what is the difference between tgt and ref?
            out_images = mpi_render_view_torch(rgba, pose, input['mpi_planes'][i], intrin_tgt, intrin_src)

        images_render.append(out_images)
        images_render = torch.cat(images_render, dim=0)
        images_tgt = input['images_tgt']
        images_tgt = images_tgt.reshape(batch_size * n_targets, *images_tgt.shape[2:])
        return images_render, images_tgt


    def validate_epoch(self):
        self.model.eval()
        loss_sum = 0
        with torch.no_grad():
            for input in self.loader_val:
                input = tensor_to_device(input, self.device)
                output = self.model(input)
                loss = self.loss(output, input)
                loss_sum += loss.item()
        return loss_sum / len(self.loader_val)


    def train(self):
        for epoch in range(self.epochs):
            start_time = time.time()
            loss_train = self.train_epoch()
            loss_val = self.validate_epoch()
            end_time = time.time()
            print(f'Epoch {epoch + 1}/{self.epochs}: loss_train={loss_train:.4f}, loss_val={loss_val:.4f}, time={end_time - start_time:.4f}s')


    def loss(self, output, input):
        images_render, images_tgt = self.render_image(output, input)
        # TODO: no croping region, is it necessary to add it?
        loss = self.vgg_loss(images_render, images_tgt)
        return loss


    def load_model(self, model_path):
        model_pathlib = Path(model_path)
        if model_pathlib.exists():
            self.model.load_state_dict(torch.load(model_path))


    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


    def create_html_viewer(self, scene_idx=0):
        self.model.eval()
        scene_loader = DataLoader(self.dset_val, batch_sampler=[[scene_idx]])
        input = tensor_to_device(next(iter(scene_loader)), self.device)

        with torch.no_grad():
            output = self.model(input)
        output = torch.sigmoid(output)
        rgba_layers = output.permute(0, 3, 4, 1, 2)

        html_viewer = Path('generated-html')
        if not html_viewer.exists():
            html_viewer.mkdir()
        
        for i in range(self.n_planes):
            file_path = 'generated-html/mpi{}.png'.format(("0" + str(i))[-2:])
            img = rgba_layers[0, :, :, i, :]    # (B, H, W, n_planes, 4)
            save_image(img, file_path)
        
        imgs_src = [get_base64_encoded_image('./generated-html/mpi{}.png'.format(("0" + str(i))[-2:])) for i in range(self.n_planes)]

        with open('./deepview-mpi-viewer-template.html', 'r') as f:
            html_template = f.read()
        
        MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in imgs_src])
        html_template = html_template.replace("const mpiSources = MPI_SOURCES_DATA;",
                                            "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

        with open('./generated-html/deepview-mpi-viewer.html', 'w') as f:
            f.write(html_template)


# trainer utils
def tensor_to_device(x, device):
    """Cast a hierarchical object to pytorch device"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        for k in list(x.keys()):
            x[k] = tensor_to_device(x[k], device)
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return type(x)(tensor_to_device(t, device) for t in x)
    else:
        raise ValueError('Wrong type !')