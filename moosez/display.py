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
from moosez import constants
import pyfiglet


def logo():
    """
    Display MOOSE logo
    :return:
    """
    print(' ')
    logo_color_code = "\033[38;5;208m"
    slogan_color_code = "\033[38;5;208m"
    result = logo_color_code + pyfiglet.figlet_format("MOOSE 2.0", font="smslant").rstrip() + "\033[0m"
    text = slogan_color_code + " A part of the ENHANCE-PET framework." + "\033[0m"
    print(result)
    print(text)
    print(' ')


def expected_modality(model_name: str) -> list:
    """
    Display expected modality for the model.
    :param model_name: The name of the model.
    :return: The expected modality for the model.
    """
    if model_name == "clin_ct_bones":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Bones")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Bones")
        return ["CT"]
    elif model_name == "clin_ct_ribs":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Ribs")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Ribs")
        return ["CT"]
    elif model_name == "clin_ct_vertebrae":
        print(
            f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Vertebral bodies")
        logging.info(
            f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Vertebral "
            f"bodies")
        return ["CT"]
    elif model_name == "clin_ct_muscles":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Muscles")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Muscles")
        return ["CT"]
    elif model_name == "clin_ct_lungs":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Lungs")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Lungs")
        return ["CT"]
    elif model_name == "clin_ct_fat":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Fat")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Fat")
        return ["CT"]
    elif model_name == "clin_ct_vessels":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Vessels")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Vessels")
        return ["CT"]
    elif model_name == "clin_ct_organs":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Organs")
        logging.info(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Organs")
        return ["CT"]
    elif model_name == "clin_pt_fdg_tumor":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: FDG-PET | Tissue of interest: Tumor")
        logging.info(
            f" Requested model: {model_name} | Imaging: Clinical | Modality: FDG-PET | Tissue of interest: Tumor")
        return ["FDG_PT"]
    elif model_name == "clin_ct_all":
        print(f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: All regions")
        logging.info(
            f" Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: All regions")
        return ["CT"]
    elif model_name == "clin_fdg_pt_ct_all":
        print(
            f" Requested model: {model_name} | Imaging: Clinical | Modality: FDG-PET-CT | Tissue of interest: All regions")
        logging.info(
            f" Requested model: {model_name} | Imaging: Clinical | Modality: FDG-PET-CT | Tissue of interest: All "
            f"regions")
        return ['FDG_PT', 'CT']
    elif model_name == "preclin_mr_all":
        print(
            f" Requested model: {model_name} | Imaging: Pre-clinical | Modality: MR | Tissue of interest: All regions")
        logging.info(f" Requested model: {model_name} | Imaging: Pre-clinical | Modality: MR | Tissue of interest: All "
                     f"regions")
        return ["MR"]
    else:
        print(" Requested model is not available. Please check the model name.")
        logging.error(" Requested model is not available. Please check the model name.")


def citation():
    """
        Display manuscript citation
        :return:
        """
    print(" CITATION:")
    print(" ")
    print(
        " Shiyam Sundar LK, Yu J, Muzik O, et al. Fully-automated, semantic segmentation of whole-body 18F-FDG PET/CT "
        "images based on data-centric artificial intelligence. J Nucl Med. June 2022.")
    print(" Copyright 2022, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")


def expectations(model_name):
    """
    Display expected modality for the model. This is used to check if the user has provided the correct modality.
    :param model_name: The name of the model.
    """
    modalities = expected_modality(model_name)
    expected_suffix = [modality + "_" for modality in modalities]
    print(f" Required modalities: {modalities} |  No. of modalities: {len(modalities)} "
          f"| Required Suffix for non-DICOM files: {expected_suffix} ")
    logging.info(f" Required modalities: {modalities} |  No. of modalities: {len(modalities)} "
                 f"| Required Suffix for non-DICOM files: {expected_suffix} ")
    # Print the original message to the console in orange color
    print(
        f"{constants.ANSI_ORANGE} Warning: Subjects which don't have the required modalities [check file suffix] "
        f"will be skipped. {constants.ANSI_RESET}")

    # Log the more detailed warning message
    warning_message = " Skipping subjects without the required modalities (check file suffix).\n" \
                      " These subjects will be excluded from analysis and their data will not be used."
    logging.warning(warning_message)
