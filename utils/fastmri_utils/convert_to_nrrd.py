# encoding: utf-8
__author__ = 'Jonas Teuwen'

import argparse
from pathlib import Path

import SimpleITK as sitk
import h5py
import numpy as np


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description='Convert reconstruction to nrrd',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'image_fn',
        type=Path,
        help='Path to image'
    )
    parser.add_argument(
        'output',
        type=Path,
        help='Folder to output to.'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with h5py.File(args.image_fn, 'r') as data:
        if 'reconstruction_esc' in data.keys():
            image = np.array(data['reconstruction_esc'])
        elif 'reconstruction_rss' in data.keys():
            image = np.array(data['reconstruction_rss'])
        else:
            image = np.array(data['reconstruction'])

    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing((0.5, 0.5, 3.0))
    sitk.WriteImage(sitk_image, str(args.output / args.image_fn.with_suffix('.nrrd').name), True)


if __name__ == '__main__':
    main()
