"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', nargs='+', default=[50, 50], type=int, help='Resolution of images')

        # Data parameters
        self.add_argument('--challenge', choices=['singlecoil', 'multicoil', 'multicoil_csm', 'multicoil_sense'],
                          required=True, default='multicoil', help='Which challenge. '
                                                                   'multicoil_csm and multicoil_sense are only '
                                                                   'applicable to the fastmri data. '
                                                                   'Choose to create coil sensitivity maps or load '
                                                                   'them stored in disk under sense dir '
                                                                   'For other data use the multicoil option.')

        self.add_argument('--data-path', type=pathlib.Path, required=True, help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        # Mask parameters
        self.add_argument('--subsampling_dist', choices=['cartesian', 'gaussian1d', 'gaussian2d', 'gaussian3d',
                                                         'radial2d', 'radial3d', 'periodic1d', 'periodic2d',
                                                         'poisson1d'], default='cartesian',
                          help='Choose subsampling distribution type.')
        self.add_argument('--accelerations', nargs='+', default=[4, 8], type=int,
                          help='Ratio of k-space columns to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for '
                               'each volume.')
        self.add_argument('--center-fractions', nargs='+', default=[0.08, 0.04], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')
        self.add_argument('--fwhms', nargs='+', default=[0.7, 0.7], type=float,
                          help='If gaussian distribution is chosen then specify FWHM value. Default 0.7')

        # add data type option
        self.add_argument('--data_type', choices=['h5', 'pickle', 'memmap'], default='h5',
                          help='Choose store format of the data.')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
