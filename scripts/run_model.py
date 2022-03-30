"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gc
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

from fastMRI.common.args import Args
from fastMRI.common.utils import save_qMRIs
from training_utils.load.data_qMRI_loaders import create_testing_sense_loaders
from training_utils.helpers import image_loss
from scripts.train_model import load_model

torch.backends.cudnn.benchmark = False


def run_rim(args, model, data_loader):
    model.eval()
    qMRI_RIM = defaultdict(list)
    qMRI_GT = defaultdict(list)
    qMRI_Init = defaultdict(list)
    for i, data in enumerate(data_loader):
        R2star_map_init, S0_map_init, B0_map_init, phi_map_init, R2star_map_target, S0_map_target, \
        B0_map_target, phi_map_target, y_ksp, mask_brain, sampling_mask, TEs, sensitivity_map, fnames, slices = data

        print('Reconstruction ' + str(i))

        TEs = TEs[0].to(args.device)

        R2star_map_init = R2star_map_init.to(args.device)
        S0_map_init = S0_map_init.to(args.device)
        B0_map_init = B0_map_init.to(args.device)
        phi_map_init = phi_map_init.to(args.device)

        R2star_map_target = R2star_map_target.to(args.device)
        S0_map_target = S0_map_target.to(args.device)
        B0_map_target = B0_map_target.to(args.device)
        phi_map_target = phi_map_target.to(args.device)

        sensitivity_map = sensitivity_map.to(args.device)
        sampling_mask = sampling_mask.to(args.device)
        mask_brain = mask_brain.to(args.device)
        y_ksp = y_ksp.to(args.device)

        # save init and groundtruch data for later comparison
        map_init = torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], 1).to('cpu')
        for i in range(map_init.shape[0]):
            qMRI_Init[fnames[0]].append((slices[i].numpy(), map_init[i].numpy()))
        map_gt = torch.stack([R2star_map_target, S0_map_target, B0_map_target, phi_map_target], 1).to('cpu')
        for i in range(map_gt.shape[0]):
            qMRI_GT[fnames[0]].append((slices[i].numpy(), map_gt[i].numpy()))

        if args.use_rim:
            estimate = model.forward(y=torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], 1), y_ksp=y_ksp, mask_subsampling=sampling_mask, mask_brain=torch.ones_like(mask_brain), TEs=TEs, sense=sensitivity_map, metadata=[])
            output = estimate[len(estimate)-1].to('cpu')
        else:
            estimate = model.forward(torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], 1))
            estimate = [_.unsqueeze(0) for _ in estimate]
            output = estimate[len(estimate)-1].to('cpu')

        del estimate, y_ksp, sensitivity_map, sampling_mask

        print('Loss of init map: ', image_loss(map_init, map_gt, mask_brain.to('cpu'), args).detach().numpy())
        print('Loss of rim map: ', image_loss(output, map_gt, mask_brain.to('cpu'), args).detach().numpy())

        for i in range(output.shape[0]):
            qMRI_RIM[fnames[0]].append((slices[i].detach().numpy(), output[i].detach().numpy()))
                
    gc.collect()
    torch.cuda.empty_cache()

    qMRI_RIM = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in qMRI_RIM.items()
    }
    qMRI_Init = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in qMRI_Init.items()
    }
    qMRI_GT = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in qMRI_GT.items()
    }

    return qMRI_RIM, qMRI_Init, qMRI_GT

def main(args):

    data_loader = create_testing_sense_loaders(args)
    checkpoint, model, optimizer = load_model(args.checkpoint)
    args.n_slices = checkpoint['args'].n_slices
    print('Reconstructing...')
    qMRI_RIM, qMRI_Init, qMRI_GT = run_rim(args, model, data_loader)
    print('Saving...')
    save_qMRIs([qMRI_RIM, qMRI_Init, qMRI_GT], args.out_dir)
    print('Done!')


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--mask-kspace', action='store_true',
                        help='Whether to apply a mask (set to True for val data and False '
                             'for test data')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the RIM model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=2, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data_parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--coil_sum_method', type=str, choices=['rss', 'sum'], default='rss',
                        help="Choose to sum coils over rss or torch.sum")
    parser.add_argument('--sequence', type=str, choices=['MEGRE', 'FUTURE_SEQUENCES'], default='ME_GRE',
                        help="Choose for which sequence to compute the parameter maps")
    parser.add_argument('--TEs', type=tuple, default=(3, 11.5, 20, 28.5), help="Echo times (/ms) in the ME_GRE sequence.")
    parser.add_argument('--loss', choices=['l1', 'mse', 'ssim'], default='mse', help='for evaluation')
    parser.add_argument('--loss_subsample', type=float, default=1., help='Sampling rate for loss mask')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
