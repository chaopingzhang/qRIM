"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# TODO: use _bart to run script.

import logging
import multiprocessing
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import bart
import h5py
import numpy as np
import torch
from utils.h5_data import H5SliceData

from fastMRI.common.args import Args
from fastMRI.data import transforms
from utils.subsample import MaskFunc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_to_complex_numpy(data):
    """
    Converts a complex pytorch tensor to a complex numpy array.
    The last axis denote the real and imaginary parts respectively.
    Parameters
    ----------
    data : torch.Tensor
        Input data
    Returns
    -------
    Complex valued np.ndarray
    """
    assert data.shape[-1] == 2, 'Last axis has to denote the complex and imaginary part and should therefore be 2.'
    # TODO: Check device and detaching from computation graph
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


def save_outputs(outputs, output_path):
    sensitivity_maps = defaultdict(list)
    for filename, slice_data, pred in outputs:
        sensitivity_maps[filename].append((slice_data, pred))
    sensitivity_maps = {
        filename: np.stack([pred for _, pred in sorted(slice_preds)])
        for filename, slice_preds in sensitivity_maps.items()
    }
    for filename in sensitivity_maps:
        output_path = (output_path / filename).with_suffix('.h5')
        logger.info(f'Writing: {output_path}...')

        sensitivity_map = sensitivity_maps[filename].transpose(0, -1, 2, 3, 1)[..., 0].astype(np.complex64)
        with h5py.File(output_path, 'w') as f:
            f['sensitivity_map'] = sensitivity_map


class DataTransform:
    def __init__(self, center_fraction, acceleration):
        self.center_fraction = center_fraction
        self.acceleration = acceleration

        if center_fraction:
            self.mask_func = MaskFunc(
                [self.center_fraction], [self.acceleration], use_masks_from_dict=False,
                mask_type='cartesian', return_uncentered=False, uniform_range=False)
            # mask_type='lines', return_uncentered=False, uniform_range=False)
        else:
            self.mask_func = None

    def __call__(self, kspace, sensitivity_map, attrs, filename, slice_no, mask=None):
        if mask is None:
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, None)
            num_cols = np.array(masked_kspace.shape)[-2]
            num_low_frequencies = int(round(num_cols * self.center_fraction))

        else:
            masked_kspace = kspace
            num_low_frequencies = attrs['num_low_frequency']

        return masked_kspace, num_low_frequencies, filename, slice_no


def create_data_loader(args, data_root):
    if args.acceleration:
        if args.acceleration == 4:
            center_fraction = 0.08
        elif args.acceleration == 8:
            center_fraction = 0.04
        else:
            raise InputError()
    else:
        center_fraction = None
    logger.info(f'Running estimation for {args.file_name}')
    data = H5SliceData(
        root=data_root,
        transform=DataTransform(center_fraction, args.acceleration),
        filter_list=[args.file_name.name],
        pass_mask=args.use_provided_masks
    )
    return data


def compute_sensitivity_maps(kspace, num_low_freqs):
    """
    Run ESPIRIT coil sensitivity algorithm using the BART toolkit.
    """
    kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
    kspace = tensor_to_complex_numpy(kspace)
    # Estimate sensitivity maps
    sens_maps = bart.bart(1, f'ecalib -d0 -m1 -r {num_low_freqs}', kspace)
    return sens_maps


def run_model(idx):
    masked_kspace, num_low_frequencies, filename, slice_no = data[idx]
    sensitivity_map = compute_sensitivity_maps(masked_kspace, num_low_frequencies)
    return filename, slice_no, sensitivity_map


def main(num_workers):
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(data)):
            outputs.append(run_model(i))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(data)))
            time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken:}s')
    save_outputs(outputs, args.output_path)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--data_root', type=Path, default=None,
                        help='Path to root')
    parser.add_argument('--file_name', type=Path, default=None,
                        help='Path to file')
    parser.add_argument('--acceleration', type=int, default=None,
                        help='Generate for acceleration')
    parser.add_argument('--output_path', type=Path, default=None,
                        help='Path to save the sensitivity maps to')
    parser.add_argument('--use-provided-masks', action='store_true',
                        help='If set, the masks as provided in the h5file will be used.')
    parser.add_argument('--num-procs', type=int, default=20,
                        help='Number of processes. Set to 0 to disable multiprocessing.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = create_data_loader(args, args.data_root)
    main(args)
