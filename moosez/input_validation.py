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
# and meet the required specifications.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform input validation.
#
# ----------------------------------------------------------------------------------------------------------------------

import os
from moosez import constants
from moosez import system


def select_moose_compliant_subjects(subject_paths: list[str], modality_tags: list[str], output_manager: system.OutputManager) -> list[str]:
    """
    Selects the subjects that have the files that have names that are compliant with the moosez.

    :param subject_paths: The path to the list of subjects that are present in the parent directory.
    :type subject_paths: List[str]
    :param modality_tags: The list of appropriate modality prefixes that should be attached to the files for
                          them to be moose compliant.
    :type modality_tags: List[str]
    :param output_manager: The output manager that will be used to write the output files.
    :type output_manager: system.OutputManager
    :return: The list of subject paths that are moose compliant.
    :rtype: List[str]
    """
    # go through each subject in the parent directory
    moose_compliant_subjects = []
    for subject_path in subject_paths:
        # go through each subject and see if the files have the appropriate modality prefixes

        files = [file for file in os.listdir(subject_path) if file.endswith('.nii') or file.endswith('.nii.gz')]
        prefixes = [file.startswith(tag) for tag in modality_tags for file in files]
        if sum(prefixes) == len(modality_tags):
            moose_compliant_subjects.append(subject_path)
    output_manager.console_update(f"{constants.ANSI_ORANGE} Number of moose compliant subjects: {len(moose_compliant_subjects)} out of {len(subject_paths)} {constants.ANSI_RESET}")
    output_manager.log_update(f" Number of moose compliant subjects: {len(moose_compliant_subjects)} out of {len(subject_paths)}")

    return moose_compliant_subjects
