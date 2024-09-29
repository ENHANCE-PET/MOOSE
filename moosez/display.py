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

import logging
import emoji
import pyfiglet
from moosez import constants
from moosez import models


def get_usage_message():
    usage_message = """
    Usage:
      moosez -d <MAIN_DIRECTORY> -m <MODEL_NAME>
    Example:  
      moosez -d /Documents/Data_to_moose/ -m clin_ct_organs

    Description:
      MOOSE (Multi-organ objective segmentation) - A data-centric AI solution that
      generates multilabel organ segmentations for systemic TB whole-person research.
    """
    return usage_message


def logo():
    """
    Display MOOSE logo

    This function displays the MOOSE logo using the pyfiglet library and ANSI color codes.

    :return: None
    """
    print(' ')
    logo_color_code = constants.ANSI_VIOLET
    slogan_color_code = constants.ANSI_VIOLET
    result = logo_color_code + pyfiglet.figlet_format(" MOOSE 3.0", font="smslant").rstrip() + "\033[0m"
    text = slogan_color_code + " A part of the ENHANCE community. Join us at www.enhance.pet to build the future of " \
                               "PET imaging together." + "\033[0m"
    print(result)
    print(text)
    print(' ')


def citation():
    """
    Display manuscript citation

    This function displays the manuscript citation for the MOOSE project.

    :return: None
    """
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":scroll:")} CITATION:{constants.ANSI_RESET}')
    print(" ")
    print(
        " Shiyam Sundar LK, Yu J, Muzik O, et al. Fully-automated, semantic segmentation of whole-body 18F-FDG PET/CT "
        "images based on data-centric artificial intelligence. J Nucl Med. June 2022.")
    print(" Copyright 2022, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")


def expectations(model_routine: dict[tuple, list[models.ModelWorkflow]]) -> list:
    """
    Display expected modality for the model.

    This function displays the expected modality for the given model name. It also checks for a special case where
    'FDG-PET-CT' should be split into 'FDG-PET' and 'CT'.

    :param model_routine: A model routine dictionary
    :type model_routine: dict
    :return: A list of modalities.
    :rtype: list
    """
    required_modalities = []
    required_prefixes = []

    for workflows in model_routine.values():
        for workflow in workflows:
            modalities, prefixes = workflow.get_expectations()
            required_modalities = required_modalities + modalities
            required_prefixes = required_prefixes + prefixes

    required_modalities = list(set(required_modalities))
    required_prefixes = list(set(required_prefixes))

    print(f" Required modalities: {required_modalities} | No. of modalities: {len(required_modalities)}"
          f" | Required prefix for non-DICOM files: {required_prefixes}")
    logging.info(f" Required modalities: {required_modalities} | No. of modalities: {len(required_modalities)} "
                 f"| Required prefix for non-DICOM files: {required_prefixes} ")
    print(f"{constants.ANSI_ORANGE} Warning: Subjects which don't have the required modalities [check file prefix] "
          f"will be skipped. {constants.ANSI_RESET}")
    logging.warning(" Skipping subjects without the required modalities (check file prefix).\n"
                    " These subjects will be excluded from analysis and their data will not be used.")

    return required_modalities
