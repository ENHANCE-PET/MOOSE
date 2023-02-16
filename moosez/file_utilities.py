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

import os
import sys
from datetime import datetime
import shutil
from multiprocessing import Pool
from moosez import constants
from halo import Halo


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
    output_dirs = []
    for modality in modalities:
        input_dirs.append(os.path.join(moose_dir, 'input_' + modality))
        create_directory(input_dirs[-1])
        output_dirs.append(os.path.join(moose_dir, 'output_' + modality))
        create_directory(output_dirs[-1])
    return moose_dir, input_dirs, output_dirs


def nnunet_folder_structure():
    """
    Creates the nnunet folder structure.
    """
    nnunet_dir = os.path.join(constants.MOOSEZ_MODEL_FOLDER)
    create_directory(nnunet_dir)
    return nnunet_dir


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
            if file.startswith(modality_tag):
                selected_files.append(os.path.join(subject, file))
    return selected_files


def organise_files_by_modality(moose_compliant_subjects: list, modalities: list, moose_dir) -> None:
    """
    Organises the files by modality.
    :param moose_compliant_subjects: The list of moose-compliant subjects paths.
    :param modalities: The list of modalities.
    :param moose_dir: The path to the moose directory.
    """
    spinner = Halo(text='Organising files by modality...', spinner='dots')
    spinner.start()
    for modality in modalities:
        files_to_copy = select_files_by_modality(moose_compliant_subjects, modality)
        copy_files_to_destination(files_to_copy, os.path.join(moose_dir, 'input_' + modality))
    spinner.succeed('Files organised by modality.')
