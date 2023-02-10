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

import pyfiglet


def logo():
    """
    Display MOOSE logo
    :return:
    """
    print("\n")
    result = pyfiglet.figlet_format("MOOSE 2.0", font="slant")
    print(result)


def extended_model_name(model_name: str):
    """
    Shows the extended model name.
    :param model_name: The name of the model.
    :return: The extended model name.
    """
    if model_name == "clin_ct_bones":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Bones"
    elif model_name == "clin_ct_ribs":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Ribs"
    elif model_name == "clin_ct_vertebrae":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Vertebrae"
    elif model_name == "clin_ct_muscles":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Muscles"
    elif model_name == "clin_ct_fat":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Fat"
    elif model_name == "clin_ct_vessels":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Vessels"
    elif model_name == "clin_ct_organs":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: Organs"
    elif model_name == "clin_pt_fdg_tumor":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: FDG PET | Tissue of interest: Tumor"
    elif model_name == "clin_ct_all":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: CT | Tissue of interest: All regions"
    elif model_name == "clin_fdg_pt_ct_all":
        return f"Requested model: {model_name} | Imaging: Clinical | Modality: FDG PET/CT | Tissue of interest: All " \
               f"regions "
    elif model_name == "preclin_mr_all":
        return f"Requested model: {model_name} | Imaging: Pre-clinical | Modality: MR | Tissue of interest: All regions"


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
