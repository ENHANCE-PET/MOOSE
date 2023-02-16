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


def check_directory_exists(directory_path: str):
    """
    Checks if the specified directory exists.
    :param directory_path: The path to the directory.
    :raises: Exception if the directory does not exist.
    """
    if not os.path.isdir(directory_path):
        raise Exception(f"Error: The directory '{directory_path}' does not exist.")


def select_moose_compliant_subjects(parent_directory: str, modality_tags: list) -> list:
    """
    Selects the subjects that have the files that have names that are compliant with the moosez.
    :param parent_directory: The path to the parent directory.
    :param modality_tags: The list of appropriate modality suffixes that should be attached to the files for them to be moose
    compliant.
    :return: The list of subject paths that are moose compliant.
    """
    # go through each subject in the parent directory
    subjects = os.listdir(parent_directory)
    moose_compliant_subjects = []
    for subject in subjects:
        # go through each subject and see if the files have the appropriate modality suffixes
        subject_path = os.path.join(parent_directory, subject)
        files = os.listdir(subject_path)
        suffixes = [file.endswith(tag) for tag in modality_tags for file in files]
        if all(suffixes):
            moose_compliant_subjects.append(os.path.join(parent_directory, subject))
    print("\033[33m" + f"Number of moose compliant subjects: {len(moose_compliant_subjects)} out of "
                       f"{len(os.listdir(parent_directory))}" + "\033[0m")
    logging.info(f"Number of moose compliant subjects: {len(moose_compliant_subjects)} out of "
                 f"{len(os.listdir(parent_directory))}")

    return moose_compliant_subjects
