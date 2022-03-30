# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.file_utils import read_json, read_list, write_list
from utils.mri_spike_noise import compute_spike_noise

from fastMRI.data import transforms


class H5SliceData(Dataset):
    """
    A PyTorch Dataset class which outputs k-space slices based on the h5 dataformat.
    """

    def __init__(
            self, root, transform, filter_list=None,
            remove_empty=False, remove_spikes=0, sensitivity_maps=None, pass_mask=False):
        """
        Initialize the dataset. The dataset can remove spike noise and empty slices.

        Parameters
        ----------
        root : Path
            Root directory to data.
        transform : func
            Function to transform the data before outputting.
        filter_list : filename or list
            Use this list of filenames to filter the possible data points in root.
        remove_empty : bool
            Remove empty slices. This might be needed to get more fair training and validation metrics.
            Some metrics also crash on empty input.
        remove_spikes : int
            0: no removal,
            1: skip,
            2: filter.
            Assumes that there is a json file 'noise_values.json' in root directory if mode 1 or 2.
        sensitivity_maps : [pathlib.Path, None]
            Path to sensitivity maps, or None
        pass_mask : bool
            Get the mask from the data file and pass to the DataTransform, useful for evaluation.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.root = pathlib.Path(root)
        self.transform = transform

        self.examples = []

        path_to_available_h5s = self.root / 'available_h5s.lst'
        if not path_to_available_h5s.exists():
            self.logger.info(f'Parsing directory {self.root}.')
            files = list(self.root.iterdir())
            files = [val for val in files if str(val)[-3:] == '.h5']
            self.logger.info('Found {len(files} files.')
            write_list(path_to_available_h5s, [_.name for _ in files])
        else:
            files = read_list(path_to_available_h5s)
            files = [self.root / _ for _ in files]

        self._spikes_dict = {}
        self.remove_spikes = remove_spikes
        if remove_spikes > 0:
            self._spikes_dict = read_json(root / 'noise_values.json')

        path_to_parsed_files = self.root / 'dataset_shapes.pkl'
        examples = []
        # This process can be split across GPUs.
        if not path_to_parsed_files.exists():
            self.logger.info(f'Parsing dataset and storing result to {path_to_parsed_files}.')
            for idx, filename in enumerate(sorted(files)):
                if len(files) % (idx + 1) == 5:
                    self.logger.info(f'({(idx + 1) / len(files) * 100:.2f}%) Parsing...')
                kspace = h5py.File(filename, 'r')['kspace']
                num_slices = kspace.shape[0]
                for slice_no in range(num_slices):
                    if remove_empty and np.abs(kspace[slice_no]).sum() == 0.0:
                        if slice_no == 0:
                            self.logger.info(f'Removing empty for {filename}...')
                        self.logger.info(f'Slice {slice_no} empty, removing from dataset.')
                        continue
                    examples.append((filename, slice_no))
            with open(path_to_parsed_files, 'wb') as f:
                pickle.dump(examples, f)
        else:
            with open(path_to_parsed_files, 'rb') as f:
                examples = pickle.load(f)
        if filter_list:
            filter_fns = read_list(filter_list if isinstance(filter_list, (list, tuple)) else root / filter_list)
        else:
            filter_fns = [_.name for _ in files]
        self.examples = [(filename, slice_no) for filename, slice_no in examples if filename.name in filter_fns]

        if self.remove_spikes == 1:
            for filename, slice_no in self.examples:
                examples = []
                spikes = self._spikes_dict[filename.name]['spike_idx']
                if slice_no in spikes:
                    self.logger.info(f'Removing slice {slice_no} of {filename} because of spike artifact.')
                else:
                    examples.append((filename, slice_no))
            self.examples = examples

        self.sensitivity_maps = sensitivity_maps
        self.pass_mask = pass_mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        filename, slice_no = self.examples[idx]
        with h5py.File(filename, 'r') as data:
            kspace = data['kspace'][slice_no]
            if not self.pass_mask:
                mask = None
            else:
                mask = data['mask'][()]
                mask = (mask * np.ones(kspace.shape).astype(np.float32))[..., np.newaxis]
                mask = torch.from_numpy(mask)

            if kspace.ndim == 2:  # Singlecoil data does not always have coils at the first axis.
                kspace = kspace[np.newaxis, ...]

            # If the spikes have to be removed, e.g. during testing, this is done here.
            if self.remove_spikes == 2:
                if slice_no in self._spikes_dict[filename.name]['spike_idx']:
                    self.logger.info(f'Removing spikes from {slice_no} of {filename}.')
                    spike_idx, _, spike_mask = compute_spike_noise(kspace)
                    kspace[spike_mask] = 0

            # If the sensitivity maps exist, load these
            if self.sensitivity_maps:
                with h5py.File(self.sensitivity_maps / filename.name, 'r') as sens:
                    sensitivity_map = sens['sensitivity_map'][slice_no]
            else:
                # If there are no sensitivity maps, generate unit sensitivity.
                sensitivity_map = np.zeros_like(kspace).astype(np.complex128)
                sensitivity_map[...] = 1. + 0 * 1j

            kspace = transforms.to_tensor(kspace).float()
            sensitivity_map = transforms.to_tensor(sensitivity_map).float()
            return self.transform(kspace, sensitivity_map, data.attrs, filename.name, slice_no, mask)
