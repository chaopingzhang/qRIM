#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:35:26 2020

@author: czhang
"""

import numpy as np
import torch
import math
import numbers
from torch import nn
from torch.nn import functional as F
from skimage.restoration import unwrap_phase
from fastMRI.data import transforms
from training_utils import helpers
from pdb import set_trace as bp
import time


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

    
class LeastSquares:
    def __init__(self):
        pass

    def lstq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """

        q, r = torch.qr(A)
        print("device of q, r, A, Y", q.device, r.device, A.device, Y.device)
        x = torch.inverse(r) @ q.permute(0, 2, 1) @ Y
        return x

    def lstq_pinv(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """

        if Y.dim() == 2:
            return torch.matmul(torch.pinverse(Y), A)
        else:
            return torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y.permute(0,2,1), Y)), Y.permute(0,2,1)), A)

    def lstq_pinv_complex_np(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        
        if Y.ndim == 2:
            return np.matmul(np.linalg.pinv(Y), A)
        else:
            return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Y.conj(), [0,2,1]), Y)), np.transpose(Y.conj(), [0,2,1])), A)


def S0_mapping_complex(image, TEs, R2star_map, B0_map):

    if not isinstance(R2star_map, np.ndarray):
        R2star_map = R2star_map.numpy()
    if not isinstance(B0_map, np.ndarray):
        B0_map = B0_map.numpy()

    ls = LeastSquares()
    sz_image = image.size()

    image_complex = image[..., 0].numpy() + 1j * image[..., 1].numpy()
    image_complex = image_complex[0:4, ...]
    image_complex_flatten = image_complex.reshape(image_complex.shape[0], -1)
    R2star_B0_complex_map = R2star_map + 1j * B0_map
    R2star_B0_complex_map_flatten = R2star_B0_complex_map.flatten()
    scaling = 1e-3
    TEs_r2 = np.expand_dims(TEs[0:4], axis=1) * - R2star_B0_complex_map_flatten
    S0_map = ls.lstq_pinv_complex_np(np.expand_dims(np.transpose(image_complex_flatten, [1, 0]), axis=2), np.exp(scaling * np.expand_dims(np.transpose(TEs_r2, [1, 0]), axis=2)))
    S0_map = S0_map[:, 0, 0].reshape(sz_image[1:-1])

    return torch.from_numpy(np.float32(S0_map.real)), torch.from_numpy(np.float32(S0_map.imag))


def R2star_B0_real_S0_complex_mapping(image, TEs, mask_brain, mask_head, fullysample):
    
    R2star_map, _ = R2star_S0_mapping(image, TEs)   
    R2star_map = torch.from_numpy(R2star_map).float()
    B0_map = -B0_phi_mapping(image, TEs, mask_brain, mask_head, fullysample)[0]
    S0_map_real, S0_map_imag = S0_mapping_complex(image, TEs, R2star_map, B0_map)
    return R2star_map, S0_map_real, B0_map, S0_map_imag


def R2star_S0_mapping(image, TEs):
    """
    R2* map and S0 map estimation for multi-echo GRE from stored magnitude image
    files acquired at multiple TEs.

    Args:
        image (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        R2star_map (torch.Tensor): The R2* map of the input images and TEs.
        S0_map (torch.Tensor): The S0 map of the input images and TEs.
    """
        
    scaling = 1e-3
    image_abs_flatten = torch.flatten(transforms.complex_abs(image), start_dim=1, end_dim=-1)
    R2star_map = np.zeros([image_abs_flatten.shape[1]])
    S0_map = np.zeros([image_abs_flatten.shape[1]])
    for i in range(image_abs_flatten.shape[1]):
        R2star_map[i], S0_map[i] = np.polyfit(TEs*scaling, np.log(image_abs_flatten[:,i]), 1, w=np.sqrt(image_abs_flatten[:,i]))
    S0_map = np.exp(S0_map)
    R2star_map = np.reshape(-R2star_map, image.shape[1:4])
    S0_map = np.reshape(S0_map, image.shape[1:4])

    return R2star_map, S0_map


def B0_phi_mapping(image, TEs, mask_brain, mask_head, fullysample):
    """
    B0 map and Phi map estimation for multi-echo GRE from stored magnitude image
    files acquired at multiple TEs.

    Args:
        image (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        BO_map (torch.Tensor): The B0 map of the input images and TEs.
        phi_map (torch.Tensor): The Phi map of the input images and TEs.
    """
    
    TEnotused = 3 if fullysample else 3
        
    # mask_brain is used only for descale of phase difference (so that phase_diff is in between -2pi and 2pi)
    mask_brain_descale = mask_brain
    sz_image = image.size()
    TEs = TEs.to(image.device)

    # phase of images
    #apply gaussian blur with radius r
    smoothing = GaussianSmoothing(2, 9, 1)
    image = image.permute([0,1,4,2,3])
    for i in range(image.shape[0]):
        # input = torch.rand(1, 3, 100, 100)
        image_tmp = F.pad(image[i], (4, 4, 4, 4), mode='reflect')
        image[i] = smoothing(image_tmp)
    image = image.permute([0,1,3,4,2])
    
    phase = torch.from_numpy(np.angle(image[..., 0].cpu().detach().numpy() + 1j * image[..., 1].cpu().detach().numpy())).float().to(image.device)
    sz_phase = phase.shape

    B0_map = torch.zeros(sz_phase[1],sz_phase[2],sz_phase[3]).to(image.device)
    phi_map = torch.zeros(sz_phase[1],sz_phase[2],sz_phase[3]).to(image.device)
    # unwrap phases (unwrap_phase doesn't support batched operation, so loop over batch size)
    for j in range(sz_phase[1]):  # loop over batch size
        phase_sample = phase[:,j,:,:].squeeze()  # since umwrap_phase accepts only 3D or 2D array
        phase_unwrapped = torch.zeros(sz_phase[0], sz_phase[2], sz_phase[3]).to(image.device)
        mask_head_np = mask_head.cpu().detach().numpy() if sz_phase[1]==1 else mask_head[0].cpu().detach().numpy()
        mask_head_np = np.invert(mask_head_np > 0.5)
        for i in range(phase_sample.shape[0]):
            phase_unwrapped[i] = torch.from_numpy(unwrap_phase(np.ma.array(phase_sample[i].squeeze().cpu().detach().numpy(), mask=mask_head_np)).data).float().to(image.device)
        phase_diff_set = []
        TE_diff = []

        # obtain phase differences and TE differences
        testplot = []
        for i in range(0, phase_unwrapped.shape[0] - TEnotused):
            phase_diff_set.append(torch.flatten(phase_unwrapped[i + 1] - phase_unwrapped[i]))
            phase_diff_set[i] = phase_diff_set[i] - torch.round(torch.sum(phase_diff_set[i]*torch.flatten(mask_brain_descale))/torch.sum(mask_brain_descale)/2/np.pi) *2*np.pi
            TE_diff.append(TEs[i + 1] - TEs[i])

        phase_diff_set = torch.stack(phase_diff_set, 0).to(image.device)
        TE_diff = torch.stack(TE_diff, 0).to(image.device)

        # least squares fitting to obtain phase map
        scaling = 1e-3
        ls = LeastSquares()
        B0_map_tmp = ls.lstq_pinv(phase_diff_set.unsqueeze(2).permute(1, 0, 2), TE_diff.unsqueeze(1) * scaling)
        B0_map[j] = B0_map_tmp[:, 0, 0].reshape(sz_image[-3], sz_image[-2]).to(image.device)
        B0_map[j] = B0_map[j] * mask_head
        # obtain phi map
        phi_map[j] = phase_unwrapped[0] - scaling * TEs[0] * B0_map[j].to(image.device)

    return B0_map, phi_map


def R2star_S0_mapping_from_ksp(kspace, TEs, sense, mask_brain, mask_head, fullysample=True, option=0):

    # insert a dimension of batch, if there is not one there already.
    # the full dims of kspace should be [nr_TEs, nr_batches, nr_coils, nr_phases, nr_slices, 2], 2 is for real and imag components.
    if kspace.dim() == 5:
        kspace = kspace.unsqueeze(1)

    TEs_size = TEs.size(0) if type(TEs) == torch.Tensor else len(TEs)
    assert kspace.size(0) == TEs_size

    # transform kspace into image
    image = transforms.ifft2(kspace)

    mask_brain_B0 = mask_brain 
    mask_brain = torch.ones_like(mask_brain)
    if mask_brain.dim() == 2:  # to recover the batch dimension if batchsize==1
        mask_brain_tmp = mask_brain.unsqueeze(0)
    else:
        mask_brain_tmp = mask_brain
    mask_brain_tmp = mask_brain_tmp.repeat(image.shape[0],image.shape[2],image.shape[-1],1,1,1).permute(0,3,1,4,5,2)
    image = image * mask_brain_tmp

    # recon the coil-combined image
    image = helpers.combine_imgs_from_multicoil(image, sense)

    if option==0:
        return R2star_B0_real_S0_complex_mapping(image, TEs, mask_brain_B0, mask_head, fullysample)


class signal_model_forward(object):
    def __call__(self, R2star_map, S0_map, B0_map, phi_map, mask_brain, TEs=torch.Tensor([3.0, 11.5, 20.0, 28.5]), sequence='MEGRE'):
        return signal_model(R2star_map, S0_map, B0_map, phi_map, mask_brain, TEs, sequence)


def signal_model(R2star_map, S0_map, B0_map, phi_map, mask_brain, TEs, sequence):
    """
    wrapped function for generating k-spaces from parameter maps
    """
    if sequence == 'MEGRE':
        return signal_model_MEGRE(R2star_map, S0_map, B0_map, phi_map, mask_brain, TEs)


def signal_model_MEGRE_nophase(R2star_map, S0_map, B0_map, phi_map, mask_brain, TEs):
    """
    generating k-spaces from parameter maps for ME-GRE sequence
    """

    scaling = 1.0e-3
    images = torch.stack([torch.stack((S0_map * torch.exp(-TEs[i] * scaling * R2star_map),
                          S0_map * torch.exp(-TEs[i] * scaling * R2star_map)), -1)
            for i in range(TEs.size(0))], 1)
    
    images[images != images] = 0
    return images


def signal_model_MEGRE(R2star_map, S0_map, B0_map, phi_map, mask_brain, TEs):
    """
    generating k-spaces from parameter maps for ME-GRE sequence
    """

    scaling = 1.0e-3
    S0_map_real = S0_map
    S0_map_imag = phi_map
    images = torch.stack([torch.stack((S0_map_real * torch.exp(-TEs[i] * scaling * R2star_map) * torch.cos(B0_map * scaling * -TEs[i]) - S0_map_imag * torch.exp(-TEs[i] * scaling * R2star_map) * torch.sin(B0_map * scaling * -TEs[i]),
                          S0_map_real * torch.exp(-TEs[i] * scaling * R2star_map) * torch.sin(B0_map * scaling * -TEs[i]) + S0_map_imag * torch.exp(-TEs[i] * scaling * R2star_map) * torch.cos(B0_map * scaling * -TEs[i])), -1)
            for i in range(TEs.size(0))], 1)
    
    images[images != images] = 0
    return images