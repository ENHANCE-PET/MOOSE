#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# This module performs input validation for the moosez. It verifies that the inputs provided by the user are valid
# and meets the required specifications.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform input validation.
#
# ----------------------------------------------------------------------------------------------------------------------

import os
from moosez import constants
import logging
import nibabel as nib
import multiprocessing


def check_directory_exists(directory_path: str):
    """
    Checks if the specified directory exists.
    :param directory_path: The path to the directory.
    :raises: Exception if the directory does not exist.
    """
    if not os.path.isdir(directory_path):
        raise Exception(f"Error: The directory '{directory_path}' does not exist.")


def select_moose_compliant_subjects(subject_paths: list, modality_tags: list) -> list:
    """
    Selects the subjects that have the files that have names that are compliant with the moosez.
    :param subject_paths: The path to the list of subjects that are present in the parent directory.
    :param modality_tags: The list of appropriate modality suffixes that should be attached to the files for them to be moose
    compliant.
    :return: The list of subject paths that are moose compliant.
    """
    # go through each subject in the parent directory
    moose_compliant_subjects = []
    for subject_path in subject_paths:
        # go through each subject and see if the files have the appropriate modality suffixes
        files = os.listdir(subject_path)
        suffixes = [file.startswith(tag) for tag in modality_tags for file in files]
        if sum(suffixes) == len(modality_tags):
            moose_compliant_subjects.append(subject_path)
    print(f"{constants.ANSI_ORANGE} Number of moose compliant subjects: {len(moose_compliant_subjects)} out of "
          f"{len(subject_paths)} {constants.ANSI_RESET}")
    logging.info(f" Number of moose compliant subjects: {len(moose_compliant_subjects)} out of "
                 f"{len(subject_paths)}")

    return moose_compliant_subjects


def check_file_for_nnunet_compatibility(filename, input_dir):
    if filename.endswith('.nii.gz') and not filename.endswith('_0000.nii.gz'):
        # if the file is not already in the desired format
        new_filename = filename.replace('.nii.gz', '_0000.nii.gz')
        os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))
    elif not filename.endswith('.gz'):
        # if the file is not compressed, compress it
        img = nib.load(os.path.join(input_dir, filename))
        new_filename = filename + '.gz'
        nib.save(img, os.path.join(input_dir, new_filename))
        os.remove(os.path.join(input_dir, filename))
        if not new_filename.endswith('_0000.nii.gz'):
            # if the file is not already in the desired format
            new_filename_with_zeroes = new_filename.replace('.nii.gz', '_0000.nii.gz')
            os.rename(os.path.join(input_dir, new_filename), os.path.join(input_dir, new_filename_with_zeroes))


def make_nnunet_compatible(input_dir: str) -> None:
    """
    Checks the files in the specified directory to ensure they comply with the nnUNet requirements. If a file does not
    comply, it is compressed (if it is not already) and renamed to include the required tag.

    Parameters:
        input_dir (str): The path to the directory containing the files to check.

    Returns:
        None
    """
    with multiprocessing.Pool() as pool:
        # Map the check_file() function to each file in the directory
        pool.starmap(check_file_for_nnunet_compatibility, [(filename, input_dir) for filename in os.listdir(input_dir)])
