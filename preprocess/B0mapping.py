#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:44:36 2020

@author: czhang
"""

# this is to create B0 map from 3D phases of 3D fully sampled images.

## delete h5 variables
import h5py
import numpy as np
import torch
from fastMRI.data import transforms
from training_utils import helpers
from training_utils.linear_mapping import LeastSquares
from pdb import set_trace as bp
import time
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt

fname = '/home/czhang/lood_storage/divh/BMEPH/Henk/STAIRS/other/Chaoping/raw_data/phases.h5'

with h5py.File(fname, 'r') as data:
    data.keys()
    phases = data['phases'][:]
    mask_brain = data['mask_brain'][:]
    phase_unwrapped = np.zeros(phases.shape)
    for i in range(phases.shape[0]):
        phase_unwrapped[i] = unwrap_phase(np.ma.array(phases[i], mask=np.zeros(phases[i].shape)))

    TEs = (3.0, 11.5, 20.0, 28.5)
    phase_diff_set = []
    TE_diff = []
    TEnotused = 3
    # obtain phase differences and TE differences
    testplot = []
    for i in range(0, phase_unwrapped.shape[0] - TEnotused):
        phase_diff_set.append((phase_unwrapped[i + 1] - phase_unwrapped[i]).flatten())
        phase_diff_set[i] = phase_diff_set[i] - np.round(np.sum(phase_diff_set[i] * mask_brain.flatten())/np.sum(mask_brain.flatten())/2/np.pi) *2*np.pi
        TE_diff.append(TEs[i + 1] - TEs[i])

    phase_diff_set = np.stack(phase_diff_set, 0)
    TE_diff = np.stack(TE_diff, 0)

    # least squares fitting to obtain phase map
    scaling = 1e-3
    ls = LeastSquares()
    B0_map_tmp = ls.lstq_pinv(torch.from_numpy(np.transpose(np.expand_dims(phase_diff_set, 2), (1, 0, 2))), torch.from_numpy(np.expand_dims(TE_diff, 1) * scaling))
    B0_map = B0_map_tmp.reshape(phase_unwrapped.shape[1:4])
    B0_map = B0_map.numpy()

    data = h5py.File(fname, 'r+')
    data.__delitem__('B0_map')
    data.create_dataset('B0_map', data=B0_map)
