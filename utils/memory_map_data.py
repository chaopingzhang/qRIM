# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import functools as ft
import os
import pathlib
import pickle
from os.path import basename

import numpy as np
import torch
from torch.utils.data import Dataset

from fastMRI.data import transforms


class SliceMemmapData(Dataset):
    def __init__(self, root, transform, sample_rate=1, n_slices=1):
        self.root = root
        self.transform = transform
        self.n_slices = n_slices
        self.reshape_imspace, self.reshape_sense = lambda x: x, lambda x: x

        if os.path.isdir(pathlib.Path(root)):
            non_empty_dirs = {p.parent for p in pathlib.Path(root).rglob('*') if p.is_file()}
            files = [list(pathlib.Path(char).iterdir()) for char in non_empty_dirs][0]
        else:
            files = list(pathlib.Path(root).iterdir())
            files = [_ for _ in files if _.name.startswith('raw_')]

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        self.files = sorted(files)

        prev_l = 0
        prev_l_sense = 0

        self.lengths = []
        self.lengths_sense = []

        for subject in self.files:
            if str(subject).split('/')[-1] == 'imspace_header':
                with open(subject, 'rb') as f:
                    shape, _, dtype, scale = pickle.load(f)
                ishape = self.reshape_imspace(shape)
                l = int(np.ceil(ishape[0] / self.n_slices))
                prev_l += l
                self.lengths.append(prev_l)

            elif str(subject).split('/')[-1] == 'sense_header':
                with open(subject, 'rb') as f:
                    shape, _, dtype, _ = pickle.load(f)
                sshape = self.reshape_sense(shape)
                l_s = int(np.ceil(sshape[0] / self.n_slices))
                prev_l_sense += l_s
                self.lengths_sense.append(prev_l_sense)

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, index):
        idx = np.digitize(index, self.lengths)
        sindex = self.lengths_sense[idx]
        modulo = self.lengths[idx]
        if idx > 0:
            index -= self.lengths[idx - 1]
            sindex -= self.lengths_sense[idx - 1]
            modulo -= self.lengths[idx - 1]
        modulo /= sindex
        sindex = int(index // modulo)

        fname = pathlib.Path('/'.join(str(self.files[idx]).split('/')[:-1]))
        target_fname = pathlib.Path(str(fname) + '/imspace')
        sense_fname = pathlib.Path(str(fname) + '/sense')

        target = self.load_memmap(target_fname, index * self.n_slices)
        sense = self.load_memmap(sense_fname, sindex * self.n_slices)

        sense = np.transpose(sense, (3, 0, 1, 2))
        divisor = np.sqrt(np.sum(np.power(sense.real, 2.) + np.power(sense.imag, 2.), 0))
        sense = np.divide(sense, divisor, out=np.zeros_like(sense), where=divisor != 0 + 0j)

        target = target * sense.conj()

        target = np.stack((target.real, target.imag), -1).astype('float32').squeeze()
        sense = np.stack((sense.real, sense.imag), -1).astype('float32').squeeze()

        return self.transform(target, sense, fname.name, sindex)

    def load_memmap(self, fname, index):
        with open(pathlib.Path(str(fname) + '_header'), 'rb') as f:
            shape, nelem, dtype, scale = pickle.load(f)
            if len(shape) > 9:
                raise ValueError("self.reshape not programmed to work on data with 10 or more dimensions")
        shape = getattr(self, f'reshape_{basename(fname)}')(shape)
        n = ft.reduce(lambda x, y: x * y, shape[1:])
        depth = int(nelem / n)
        crop = self.n_slices if (self.n_slices + index) < depth else depth - index
        memmap = np.memmap(fname, dtype=dtype, mode='r', shape=(crop, *shape[1:]), offset=dtype.itemsize * index * n)
        out = np.copy(memmap) / scale
        del memmap
        return out


class TrainingMemmapTransform:
    def __init__(self, mask_func, resolution, train_resolution=None, use_seed=True):
        self.mask_func = mask_func
        self.resolution = resolution
        self.train_resolution = train_resolution
        self.use_seed = use_seed

    def __call__(self, target, sense, fname, slice):
        seed = None if not self.use_seed else tuple(map(ord, fname))
        np.random.seed(seed)

        target = transforms.to_tensor(target)
        kspace = transforms.fft2(target)
        target = torch.sum(transforms.complex_abs(target), 0)
        sense = transforms.to_tensor(sense)

        if self.train_resolution is not None:
            kspace = transforms.ifft2(kspace)
            p = max(kspace.size(-3) - self.train_resolution[0], kspace.size(-2) - self.train_resolution[1]) // 2 + 1
            kspace = torch.nn.functional.pad(input=kspace, pad=(0, 0, p, p, p, p), mode='constant', value=0)
            kspace = transforms.complex_center_crop(kspace, self.train_resolution)
            kspace = transforms.fft2(kspace)
            target = transforms.center_crop(target, self.train_resolution)
            sense = transforms.complex_center_crop(sense, self.train_resolution)

        # Apply mask
        if self.mask_func is not None:
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = kspace != 0
            mask = mask.to(kspace)[..., :1, :, :1]
            mask = mask[:1, ...]

        attrs = torch.tensor([])

        return transforms.ifft2(masked_kspace), mask, target, sense, attrs


class TestingMemmapTransform:
    def __init__(self, mask_func=None):
        self.mask_func = mask_func

    def __call__(self, target, sense, fname, slice):
        kspace = transforms.fft2(transforms.to_tensor(target))
        sense = transforms.to_tensor(sense)

        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = kspace != 0
            mask = mask.to(kspace)[..., :1, :, :1]
            mask = mask[:1, ...]

        if masked_kspace.dim() == 5:
            masked_kspace = masked_kspace.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return transforms.ifft2(masked_kspace), mask, sense, torch.tensor([]), fname, slice
