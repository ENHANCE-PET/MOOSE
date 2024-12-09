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
from typing import Tuple, List, Dict
from moosez import constants
from moosez import models
from moosez import system


def determine_model_expectations(model_routine: Dict[Tuple, List[models.ModelWorkflow]], output_manager: system.OutputManager) -> List:
    """
    Display expected modality for the model.

    This function displays the expected modality for the given model name. It also checks for a special case where
    'FDG-PET-CT' should be split into 'FDG-PET' and 'CT'.

    :param model_routine: The model routine
    :type model_routine: Dict[Tuple, List[models.ModelWorkflow]]
    :param output_manager: The output manager
    :type output_manager: system.OutputManager
    :return: A list of modalities.
    :rtype: list
    """
    required_modalities = []
    required_prefixes = []

    header = ["Nr", "Model Name", "Indices & Regions", "Imaging", "Required Modality", "Required Prefix (non-DICOM)",
              "Nr of training datasets"]
    styles = [None, "cyan", None, None, None, None, None]
    table = output_manager.create_table(header, styles)

    model_nr = 0
    for model_workflows in model_routine.values():
        for model_workflow in model_workflows:
            model_nr += 1
            modalities, prefixes = model_workflow.target_model.get_expectation()
            required_modalities = required_modalities + modalities
            required_prefixes = required_prefixes + prefixes

            model_identifier = model_workflow.target_model.model_identifier
            modality = model_workflow.target_model.modality
            imaging = f"{model_workflow.target_model.imaging_type}ical".capitalize()
            regions = str(model_workflow.target_model.organ_indices)
            nr_training_data = model_workflow.target_model.nr_training_data
            table.add_row(str(model_nr), model_identifier, regions, imaging, modality, ', '.join(prefixes), nr_training_data)

    output_manager.console_update(table)

    required_modalities = list(set(required_modalities))
    required_prefixes = list(set(required_prefixes))

    output_manager.log_update(f" Required modalities: {required_modalities} | No. of modalities: {len(required_modalities)} "
                              f"| Required prefix for non-DICOM files: {required_prefixes} ")
    output_manager.console_update(f"{constants.ANSI_ORANGE} Warning: Subjects which don't have the required modalities [check file prefix] "
                                  f"will be skipped. {constants.ANSI_RESET}")
    output_manager.log_update(" Skipping subjects without the required modalities (check file prefix).\n"
                              " These subjects will be excluded from analysis and their data will not be used.")

    return required_modalities


def select_moose_compliant_subjects(subject_paths: List[str], modality_tags: List[str], output_manager: system.OutputManager) -> List[str]:
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
