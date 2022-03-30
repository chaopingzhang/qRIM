#loop over all subjects
    # process data for one subject:
        ## 1. load one subject: coil images, sensitivity maps.
        ## 2. create 3D B0 map: a. obtain 3D phase; b. unwrap phase; c. compute B0 map (use first 2 echoes).
        ## 3. create 2D kspaces. a. saggital; b. axial; c. coronal.
        ## 4. load brain mask. 
        ## 5. save as .h5 for each slice in each plane of the 3 orientations.
            # loop over all orientations.
                # loop over all slices in one orientation.

import os
import numpy as np
import h5py
import glob
import torch
import time
from pathlib import Path
import SimpleITK as sitk
from matplotlib import pyplot as plt
from pdb import set_trace as bp
from training_utils.helpers import combine_imgs_from_multicoil
from training_utils.linear_mapping import LeastSquares
from skimage.restoration import unwrap_phase




def loaddata(subjectID, datapath):
    
    sense_complex = False
    coilimgs = False
    brain_mask = False 
    folders = glob.glob(datapath + 'Subcortex_'+str(subjectID).zfill(4)+'*_R02') 
    if folders:
        filename_sense = glob.glob(os.path.join(folders[0], 'Subcortex_'+str(subjectID).zfill(4)+'*_R02_inv2_rcal.mat'))
        file_coilimgs_p1 = 'Subcortex_'+str(subjectID).zfill(4)+'*_R02_inv2_' 
        file_coilimgs_p2 = '_gdataCorrected.nii.gz'
        filename_coilimgs = glob.glob(os.path.join(folders[0], file_coilimgs_p1+'*'+file_coilimgs_p2))
        if filename_sense and filename_coilimgs:
            # load sensitivity map (complex-valued)
            with h5py.File(filename_sense[0], 'r') as f:
                for k, v in f.items():
                    sense = np.array(v)
                    sense_complex =sense['real'] +1j*sense['imag']
                    sense_complex = np.transpose(sense_complex, (3,2,1,0))

            # load coil images (complex-valued)
            coilimgs = []
            for i in range(1, 5):
                filename_coilimg = glob.glob(os.path.join(folders[0], file_coilimgs_p1 + str(i) + file_coilimgs_p2))
                coilimg = sitk.ReadImage(filename_coilimg[0])
                coilimgs.append(np.transpose(sitk.GetArrayFromImage(coilimg), (3,2,1,0)))    
        
            # load brain mask
            brain_mask = sitk.ReadImage(os.path.join(folders[0], 'nii', 'mask_inv2_te2_m_corr.nii'))
            brain_mask = sitk.GetArrayFromImage(brain_mask)
            brain_mask = np.flip(np.transpose(brain_mask, (0, 2, 1)), 1)  # need to flip! in the second axis!
        
    return coilimgs, sense_complex, brain_mask


def combine_imgs_from_multicoil_np(image, sense):

    return np.sum(image*sense.conj(), -1)


def B0mapping(coilimgs, sense, mask_brain):
    
    TEnotused = 3
    
    imgs = combine_imgs_from_multicoil_np(coilimgs, sense)
    
    phases = np.angle(imgs)
    phase_unwrapped = np.zeros(phases.shape)
    for i in range(phases.shape[0]):
        phase_unwrapped[i] = unwrap_phase(np.ma.array(phases[i], mask=np.zeros(phases[i].shape)))

    TEs = (3.0, 11.5, 20.0, 28.5)
    phase_diff_set = []
    TE_diff = []
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

    return B0_map


def generate_2dksp(images3d, dim2keep):
    
    axes = [[2,3], [1,3], [1,2]]
    return np.fft.fftshift(np.fft.fft2(images3d, axes = axes[dim2keep], norm="ortho"), axes = axes[dim2keep])

if __name__ == '__main__':
    
    datapath = '/data/projects/ahead/raw2/'
    applymask = False
    if applymask:
        savepath = '/data/projects/recon/data/qMRI/Brain_MEGRE_masked/'
    else:
        centerslices = True
        if centerslices:
            savepath = '/data/projects/recon/data/qMRI/Brain_MEGRE_centerslices/'
            half_nr_of_slices = 25;
        else:
            savepath = '/data/projects/recon/data/qMRI/Brain_MEGRE/'
            half_nr_of_slices = 50;
            
    for subjectID in range(1, 119):
        start_subject = time.perf_counter()
        print(subjectID)
        coilimgs, sense, brain_mask = loaddata(subjectID, datapath)
        if coilimgs!=False:
            coilimgs = np.stack(coilimgs, axis=0)
            if applymask:
                coilimgs = coilimgs*np.repeat(brain_mask[..., np.newaxis], coilimgs.shape[-1], axis=3)
                sense = sense*np.repeat(brain_mask[..., np.newaxis], sense.shape[-1], axis=3)            
            B0map = B0mapping(coilimgs, sense, brain_mask)
            planes = ['sagittal', 'coronal', 'axial']
            folder_subject = 'Subcortex_'+str(subjectID).zfill(4)+'_R02_inv2'
            for dim in range(3):
                ksp = generate_2dksp(coilimgs, dim)
                ksp_dim = np.swapaxes(ksp, 1, dim+1)
                sense_dim = np.swapaxes(sense, 0, dim)
                B0map_dim = np.swapaxes(B0map, 0, dim)
                brain_mask_dim = np.swapaxes(brain_mask, 0, dim)
                size_dim = coilimgs.shape[dim+1]
                Path(os.path.join(savepath, folder_subject, planes[dim])).mkdir(parents=True, exist_ok=True)
                for itr_dim in range(round(size_dim/2) - half_nr_of_slices, round(size_dim/2) +half_nr_of_slices):
                    filename_save = os.path.join(savepath, folder_subject, planes[dim], 'Subcortex_'+str(subjectID).zfill(4)+'_'+planes[dim]+'_'+str(itr_dim)+'.h5')
                    with h5py.File(filename_save, 'w') as data:
                        data.create_dataset('ksp', data=ksp_dim[:,itr_dim,...].squeeze())
                        data.create_dataset('sense', data=sense_dim[itr_dim,...].squeeze())
                        data.create_dataset('B0map', data=B0map_dim[itr_dim,...].squeeze())
                        data.create_dataset('mask_brain', data=brain_mask_dim[itr_dim,...].squeeze())
        print( "%.4fs" % (time.perf_counter() - start_subject))