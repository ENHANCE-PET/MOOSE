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


def create_directory(directory_path: str):
    """
    Creates a directory at the specified path.
    :param directory_path: The path to the directory.
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def get_virtual_env_root():
    python_exe = sys.executable
    virtual_env_root = os.path.dirname(os.path.dirname(python_exe))
    return virtual_env_root


def get_files(directory, wildcard):
    """
    Returns the list of files in the directory with the specified wildcard.
    :param directory: The directory path.
    :param wildcard: The wildcard to be used.
    :return: The list of files.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(wildcard):
            files.append(os.path.join(directory, file))
    return files


def moose_folder_structure(parent_directory: str, model_name: str, modalities: list):
    """
    Creates the moose folder structure.
    :param parent_directory: The path to the parent directory.
    :param model_name: The name of the model.
    :param modalities: The list of modalities.
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


def copy_file(file, destination):
    shutil.copy(file, destination)


def copy_files_to_destination(files: list, destination: str):
    """
    Copies the files inside the list to the destination directory in a parallel fashion.
    :param files: The list of files to be copied.
    :param destination: The path to the destination directory.
    """
    with Pool(processes=len(files)) as pool:
        pool.starmap(copy_file, [(file, destination) for file in files])


def select_files_by_modality(moose_compliant_subjects: list, modality_tag: str) -> list:
    """
    Selects the files with the selected modality tag from the moose-compliant folders.
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :param modality_tag: The modality tag to be selected.
    :return: The list of selected files.
    """
    selected_files = []
    for subject in moose_compliant_subjects:
        files = os.listdir(subject)
        for file in files:
            if file.startswith(modality_tag) and (file.endswith('.nii') or file.endswith('.nii.gz')):
                selected_files.append(os.path.join(subject, file))
    return selected_files


def organise_files_by_modality(moose_compliant_subjects: list, modalities: list, moose_dir) -> None:
    """
    Organises the files by modality.
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :param modalities: The list of modalities.
    :param moose_dir: The path to the moose directory.
    """
    for modality in modalities:
        files_to_copy = select_files_by_modality(moose_compliant_subjects, modality)
        copy_files_to_destination(files_to_copy, os.path.join(moose_dir, modality))


def find_pet_file(folder):
    # Searching for files with 'PET' in their name and ending with either .nii or .nii.gz
    pet_files = glob.glob(os.path.join(folder, 'PET*.nii*')) # Files should start with PET

    if len(pet_files) == 1:
        return pet_files[0]
    elif len(pet_files) > 1:
        raise ValueError("More than one PET file found in the directory.")
    else:
        return None
