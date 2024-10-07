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
# This module shows predefined display messages for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to show predefined display
# messages.
#
# ----------------------------------------------------------------------------------------------------------------------

import emoji
import pyfiglet
import random
from moosez import constants
from moosez import models
from moosez import system


def logo(output_manager: system.OutputManager):
    """
    Display MOOSE logo

    This function displays the MOOSE logo using the pyfiglet library and ANSI color codes.

    :return: None
    """
    output_manager.console_update(' ')
    result = constants.ANSI_VIOLET + pyfiglet.figlet_format(" MOOSE 3.0", font="smslant").rstrip() + constants.ANSI_RESET
    text = constants.ANSI_VIOLET + " A part of the ENHANCE community. Join us at www.enhance.pet to build the future of " \
                               "PET imaging together." + constants.ANSI_RESET
    output_manager.console_update(result)
    output_manager.console_update(text)
    output_manager.console_update(' ')


def authors(output_manager: system.OutputManager):
    """
    Display manuscript citation

    This function displays the manuscript citation for the MOOSE project.

    :return: None
    """
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":desktop_computer:")}  AUTHORS:{constants.ANSI_RESET}')
    output_manager.console_update(" ")
    output_manager.console_update(" The Three Moose-keteers 🤺: Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer | Manuel Pires")
    output_manager.console_update(" ")

def doi(output_manager: system.OutputManager):
    """
    Display manuscript citation

    This function displays the manuscript citation for the MOOSE project.

    :return: None
    """
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":scroll:")} CITATION:{constants.ANSI_RESET}')
    output_manager.console_update(" ")
    output_manager.console_update(
        " 10.2967/jnumed.122.264063")
    output_manager.console_update(" ")
    output_manager.console_update(" Copyright 2022, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")


def expectations(model_routine: dict[tuple, list[models.ModelWorkflow]], output_manager: system.OutputManager) -> list:
    """
    Display expected modality for the model.

    This function displays the expected modality for the given model name. It also checks for a special case where
    'FDG-PET-CT' should be split into 'FDG-PET' and 'CT'.

    :param model_routine: The model routine
    :type model_routine: dict[tuple, list[models.ModelWorkflow]]
    :param output_manager: The output manager
    :type output_manager: system.OutputManager
    :return: A list of modalities.
    :rtype: list
    """
    required_modalities = []
    required_prefixes = []

    header = ["Nr", "Model Name", "Indices & Regions", "Imaging", "Required Modality", "Required Prefix (non-DICOM)"]
    styles = [None, "cyan", None, None, None, None]
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
            table.add_row(str(model_nr), model_identifier, regions, imaging, modality, ', '.join(prefixes))

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
