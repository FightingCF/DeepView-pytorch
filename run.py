# the 2 lines below are used to aviod error when launching matplotlib in more than one script
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
from trainer import Trainer

import torch
torch.manual_seed(0)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_dir', type=str, default='spaces_dataset', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs')
    parser.add_argument('--downscale_factor', type=float, default=0.5, help='the mpi size is orig_size * downscale_factor')
    parser.add_argument('--n_planes', type=int, default=12, help='number of mpi planes')
    parser.add_argument('--load_model', type=bool, default=False, help='a flag to load model or not')
    parser.add_argument('--model_path', type=str, default='ckpts/model.pth', help='path to model')

    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = get_opts()
    
    trainer = Trainer(dset_dir=opts.dset_dir, batch_size=opts.batch_size, 
                      lr=opts.lr, epochs=opts.epochs, downscale_factor=opts.downscale_factor,
                      n_planes=opts.n_planes, device='cuda')
    
    if opts.load_model:
        trainer.load_model(opts.model_path)

    trainer.train()

    trainer.create_html_viewer()