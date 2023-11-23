#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
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
import sys
from datetime import datetime
from multiprocessing import Pool

from moosez import constants


def create_directory(directory_path: str) -> None:
    """
    Creates a directory at the specified path.
    
    :param directory_path: The path to the directory.
    :type directory_path: str
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def get_virtual_env_root() -> str:
    """
    Returns the root directory of the virtual environment.
    
    :return: The root directory of the virtual environment.
    :rtype: str
    """
    python_exe = sys.executable
    virtual_env_root = os.path.dirname(os.path.dirname(python_exe))
    return virtual_env_root


def get_files(directory: str, wildcard: str) -> list:
    """
    Returns the list of files in the directory with the specified wildcard.
    
    :param directory: The directory path.
    :type directory: str
    
    :param wildcard: The wildcard to be used.
    :type wildcard: str
    
    :return: The list of files.
    :rtype: list
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(wildcard):
            files.append(os.path.join(directory, file))
    return files


def moose_folder_structure(parent_directory: str, model_name: str, modalities: list) -> tuple:
    """
    Creates the moose folder structure.
    
    :param parent_directory: The path to the parent directory.
    :type parent_directory: str
    
    :param model_name: The name of the model.
    :type model_name: str
    
    :param modalities: The list of modalities.
    :type modalities: list
    
    :return: A tuple containing the paths to the moose directory, input directories, output directory, and stats directory.
    :rtype: tuple
    """
    moose_dir = os.path.join(parent_directory,
                             'moosez-' + model_name + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    create_directory(moose_dir)
    input_dirs = []
    for modality in modalities:
        input_dirs.append(os.path.join(moose_dir, modality))
        create_directory(input_dirs[-1])

    output_dir = os.path.join(moose_dir, constants.SEGMENTATIONS_FOLDER)
    stats_dir = os.path.join(moose_dir, constants.STATS_FOLDER)
    create_directory(output_dir)
    create_directory(stats_dir)
    return moose_dir, input_dirs, output_dir, stats_dir


def copy_file(file: str, destination: str) -> None:
    """
    Copies a file to the specified destination.
    
    :param file: The path to the file to be copied.
    :type file: str
    
    :param destination: The path to the destination directory.
    :type destination: str
    """
    shutil.copy(file, destination)


def copy_files_to_destination(files: list, destination: str) -> None:
    """
    Copies the files inside the list to the destination directory in a parallel fashion.
    
    :param files: The list of files to be copied.
    :type files: list
    
    :param destination: The path to the destination directory.
    :type destination: str
    """
    with Pool(processes=len(files)) as pool:
        pool.starmap(copy_file, [(file, destination) for file in files])


def select_files_by_modality(moose_compliant_subjects: list, modality_tag: str) -> list:
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


def organise_files_by_modality(moose_compliant_subjects: list, modalities: list, moose_dir: str) -> None:
    """
    Organises the files by modality.
    
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :type moose_compliant_subjects: list
    
    :param modalities: The list of modalities.
    :type modalities: list
    
    :param moose_dir: The path to the moose directory.
    :type moose_dir: str
    """
    for modality in modalities:
        files_to_copy = select_files_by_modality(moose_compliant_subjects, modality)
        copy_files_to_destination(files_to_copy, os.path.join(moose_dir, modality))


def find_pet_file(folder: str) -> str:
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
