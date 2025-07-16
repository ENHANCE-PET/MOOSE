#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module contains the functions for performing file operations for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform file operations.
#
# ----------------------------------------------------------------------------------------------------------------------

import os
from datetime import datetime
from typing import Union, Tuple, List
from moosez import constants


def create_directory(directory_path: str) -> None:
    """
    Creates a directory at the specified path.
    
    :param directory_path: The path to the directory.
    :type directory_path: str
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def get_files(directory: str, prefix: Union[str, Tuple[str, ...]], suffix: Union[str, Tuple[str, ...]]) -> List[str]:
    """
    Returns the list of files in the directory with the specified wildcard.
    :param directory: The directory path.
    :type directory: str
    :param suffix: valid suffixes.
    :type suffix: str
    :param prefix: valid prefixes.
    :type prefix: str
    :return: The list of files.
    :rtype: list
    """

    if isinstance(prefix, str):
        prefix = (prefix,)

    if isinstance(suffix, str):
        suffix = (suffix,)

    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    modality_files = []
    for file in files:
        if file.startswith(prefix) and file.endswith(suffix):
            modality_files.append(os.path.join(directory, file))
    return modality_files


def get_modality_file(directory: str, modality_prefix: str) -> Union[str, None]:
    """
    Finds the modality file in the specified folder.
    :param directory: The path to the folder.
    :type directory: str
    :param modality_prefix: The modality prefix.
    :type modality_prefix: str
    :return: The path to the modality file.
    :rtype: str
    """

    modality_files = get_files(directory, modality_prefix, ('.nii', '.nii.gz'))

    if len(modality_files) == 1:
        return modality_files[0]
    elif len(modality_files) > 1:
        raise ValueError("More than one modality file found in the directory.")
    else:
        return None


def moose_folder_structure(parent_directory: str) -> Tuple[str, str, str]:
    """
    Creates the moose folder structure.
    
    :param parent_directory: The path to the parent directory.
    :type parent_directory: str
    
    :return: A tuple containing the paths to the moose directory, output directory, and stats directory.
    :rtype: tuple
    """
    moose_dir = os.path.join(parent_directory, 'moosez-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    create_directory(moose_dir)

    segmentation_dir = os.path.join(moose_dir, constants.SEGMENTATIONS_FOLDER)
    stats_dir = os.path.join(moose_dir, constants.STATS_FOLDER)
    create_directory(segmentation_dir)
    create_directory(stats_dir)
    return moose_dir, segmentation_dir, stats_dir


def get_nifti_file_stem(file_path: str) -> str:
    file_stem = os.path.basename(file_path).split('.gz')[0].split('.nii')[0]
    return file_stem
