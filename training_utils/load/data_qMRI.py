import pathlib
import random
import torch
import numpy as np

import h5py
from torch.utils.data import Dataset
from pdb import set_trace as bp
from fastMRI.data import transforms


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to volumes of the fully sampled
    k-spaces, the sensitivity maps, and the B0 map.
    """

    def __init__(self, root, transform, sequence, TEs, sample_rate=1, n_slices=1, use_rss=True, use_seed=True):
        """
        :param root: Path to the dataset
        :param transform: A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
        :param sequence: "ME_GRE" or "FUTURE_SEQUENCES_TO_BE_IMPLEMENTED" depending on which sequence the mapping is performed with.
        :TEs: [tuple] Echo times in "ME_GRE" sequence.
        :param sample_rate: A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        :param n_slices: Number of slices in a volume. default: 1
        :param use_rss: If True, uses RSS images as targets, also in the singlecoil case
        """
        if sequence not in ('MEGRE', 'FUTURE_SEQUENCES'):
            raise ValueError('sequence should be either "MEGRE" or "FUTURE_SEQUENCES"')

        self.TEs = TEs
        self.transform = transform
        self.maps = ['R2star_map_model', 'S0_map_model', 'B0_map_model']
        self.use_seed = use_seed

        self.examples = []
        self.n_slices = n_slices
        self.kspace_all = []
        self.plane = "axial"
        files = list(root.rglob("*"))
        # please provide accordingly the filenames of the data.
        files = [_ for _ in files if _.name.endswith('.h5') and self.plane in _.name and "maps" not in _.name and "kspmask" not in _.name and "rim_recon" not in str(_.parents[0]) and "SR_Pyqmri" not in str(_) and "modelBasedCS" not in str(_) and "cs_" not in str(_)]
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        
        self.centerlines = True
        for fname in sorted(files):
            if self.centerlines:
                str_tmp = str(fname).replace('.', '_')
                nums = [int(s) for s in str_tmp.split('_') if s.isdigit()]
                axialcenterslices = 'axial' in str(fname) and nums[-1] >= 121 and nums[-1] < 172 -1 
                coronalcenterslices = 'coronal' in str(fname) and nums[-1] >= 120 and nums[-1] < 171 -1
                sagittalcenterslices = 'sagittal' in str(fname) and nums[-1] >= 92 and nums[-1] < 143 -1
            else:
                axialcenterslices = True
                coronalcenterslices = True
                sagittalcenterslices = True
                
            if axialcenterslices or coronalcenterslices or sagittalcenterslices:
                num_slices = 1
                if n_slices == 1:
                    self.examples += [(fname, slice) for slice in range(num_slices)]
                elif n_slices == 0:
                    self.examples += [(fname, 0)]
                else:
                    self.examples += [(fname, range(num_slices))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        kspace = []
        kspname = 'ksp'
        with h5py.File(fname, 'r') as data:
            kspace = data[kspname]
            kspace = np.array(kspace)
            kspace = np.stack((kspace.real, kspace.imag), -1)
            kspace = transforms.to_tensor(kspace)
            kspace = kspace.permute(0,3,1,2,4)  # [nr_TEs, channel, phase, slice, 2]
            scalingfactor = 10000
            kspace = kspace/scalingfactor

            sensitivity_map = data['sense']
            sensitivity_map = np.array(sensitivity_map)
            sensitivity_map = np.stack((sensitivity_map.real, sensitivity_map.imag), -1) 
            sensitivity_map = np.float32(sensitivity_map)
            sensitivity_map = transforms.to_tensor(sensitivity_map)
            sensitivity_map = sensitivity_map.permute(2,0,1,3)# [channel, phase, slice, 2]

            mask_brain = data['mask_brain']
            mask_brain = np.array(mask_brain)
            mask_brain = np.float32(mask_brain)
            mask_brain = transforms.to_tensor(mask_brain)

        return self.transform(kspace, sensitivity_map, mask_brain, fname.name, fname, slice, n_slices=self.n_slices)