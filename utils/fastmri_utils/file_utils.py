# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess
import zipfile

logger = logging.getLogger(__name__)


def link_data(source_path, target_path, copy=False, is_directory=True):
    # TODO: Convert to pathlib
    if not copy:
        logger.info(f'Symlinking from {source_path} to {target_path}.')
        try:
            os.symlink(source_path, target_path, target_is_directory=is_directory)
        except FileExistsError:
            if os.path.normpath(os.path.realpath(target_path)) == os.path.normpath(source_path):
                logger.info(f'Symlink from {source_path} to {target_path} already exists.')
            else:
                raise IOError('Symlink to {target_path} exists, but does not refer to {source_path}.')

    else:
        logger.info('Copying from {} to {}.'.format(source_path, target_path))
        if is_directory:
            shutil.copytree(source_path, target_path)
        else:
            shutil.copyfile(source_path, target_path)


def read_json(fn):
    """
    Read file and output dict, or take dict and output dict

    Parameters
    ----------
    fn : list or str
        Input text file or list

    Returns
    -------
    dict
    """
    if isinstance(fn, dict):
        return fn

    with open(fn, 'r') as f:
        data = json.load(f)
    return data


def write_json(fn, data):
    """
    Write dict data to fn

    Parameters
    ----------
    fn : Path or str
    data : dict

    Returns
    -------
    None
    """
    with open(fn, 'w') as f:
        json.dump(data, f)


def read_list(fn):
    """
    Read file and output list, or take list and output list

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list

    Returns
    -------
    list
    """
    if isinstance(fn, (pathlib.Path, str)):
        with open(fn) as f:
            filter_fns = f.readlines()
        return [_.strip() for _ in filter_fns]
    return fn


def write_list(fn, data):
    """
    Write list line by line to file

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list
    data : list or tuple
    Returns
    -------
    None
    """
    with open(fn, 'w') as f:
        for line in data:
            f.write(f'{line}\n')


def unzip(source_filename, dest_dir):
    logger.info(f'Unzipping {source_filename} to {dest_dir}.')
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)


def git_hash(dir_to_git):
    """
    Get the current git hash

    Returns
    -------
    The git hash, otherwise None.

    """
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-list', '-1', 'HEAD', './'], cwd=dir_to_git).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        git_hash = None
    git_hash = git_hash if git_hash else 'no git repository found'
    logger.info(f'Current git hash is: {ghash}.')
    return git_hash
