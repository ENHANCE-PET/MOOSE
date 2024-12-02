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

import glob
import os
import shutil
from datetime import datetime
from multiprocessing import Pool
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
    
    :param suffix: The wildcard to be used.
    :type suffix: str

    :param prefix: The wildcard to be used.
    :type prefix: str
    
    :return: The list of files.
    :rtype: list
    """

    if isinstance(prefix, str):
        prefix = (prefix,)

    if isinstance(suffix, str):
        suffix = (suffix,)

    files = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(suffix):
            files.append(os.path.join(directory, file))
    return files


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


def copy_file(file: str, destination: str) -> None:
    """
    Copies a file to the specified destination.
    
    :param file: The path to the file to be copied.
    :type file: str
    
    :param destination: The path to the destination directory.
    :type destination: str
    """
    shutil.copy(file, destination)


def copy_files_to_destination(files: List[str], destination: str) -> None:
    """
    Copies the files inside the list to the destination directory in a parallel fashion.
    
    :param files: The list of files to be copied.
    :type files: list
    
    :param destination: The path to the destination directory.
    :type destination: str
    """
    with Pool(processes=len(files)) as pool:
        pool.starmap(copy_file, [(file, destination) for file in files])


def select_files_by_modality(moose_compliant_subjects: List[str], modality_tag: str) -> List:
    """
    Selects the files with the selected modality tag from the moose-compliant folders.
    
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :type moose_compliant_subjects: list
    
    :param modality_tag: The modality tag to be selected.
    :type modality_tag: str
    
    :return: The list of selected files.
    :rtype: list
    """
    selected_files = []
    for subject in moose_compliant_subjects:
        files = os.listdir(subject)
        for file in files:
            if file.startswith(modality_tag) and (file.endswith('.nii') or file.endswith('.nii.gz')):
                selected_files.append(os.path.join(subject, file))
    return selected_files


def find_pet_file(folder: str) -> Union[str, None]:
    """
    Finds the PET file in the specified folder.
    
    :param folder: The path to the folder.
    :type folder: str
    
    :return: The path to the PET file.
    :rtype: str
    """
    # Searching for files with 'PET' in their name and ending with either .nii or .nii.gz
    pet_files = glob.glob(os.path.join(folder, 'PT*.nii*'))  # Files should start with PET

    if len(pet_files) == 1:
        return pet_files[0]
    elif len(pet_files) > 1:
        raise ValueError("More than one PET file found in the directory.")
    else:
        return None


def get_nifti_file_stem(file_path: str) -> str:
    file_stem = os.path.basename(file_path).split('.gz')[0].split('.nii')[0]
    return file_stem
