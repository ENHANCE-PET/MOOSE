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
# This module contains the urls and filenames of the models and binaries that are required for the moosez.
#
# Usage:
# The variables in this module can be imported and used in other modules within the moosez to download the necessary
# binaries and models for the moosez.
#
# ----------------------------------------------------------------------------------------------------------------------
import torch

from moosez import constants

MODELS = {
    "clin_ct_bones": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/Task201_CT_Bones.zip",
        "filename": "Task201_CT_Bones.zip",
        "directory": "Task201_CT_Bones",
    },
    "clin_ct_ribs": {
        "url": "https://example.com/bones_model.zip",
        "filename": "clin_ct_ribs_model.zip",
        "directory": "clin_ct_ribs_model",
    },
    "clin_ct_vertebrae": {
        "url": "https://example.com/vertebrae_model.zip",
        "filename": "clin_ct_vertebrae_model.zip",
        "directory": "clin_ct_vertebrae_model",
    },
    "clin_ct_muscles": {
        "url": "https://example.com/muscles_model.zip",
        "filename": "clin_ct_muscles_model.zip",
        "directory": "clin_ct_muscles_model",
    },
    "clin_ct_lungs": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/clin_ct_lungs_24062023.zip",
        "filename": "Dataset333_HMS3dlungs.zip",
        "directory": "Dataset333_HMS3dlungs",
    },
    "clin_ct_fat": {
        "url": "https://example.com/fat_model.zip",
        "filename": "clin_ct_fat_model.zip",
        "directory": "clin_ct_fat_model",
    },
    "clin_ct_vessels": {
        "url": "https://example.com/vessels_model.zip",
        "filename": "clin_ct_vessels_model.zip",
        "directory": "clin_ct_vessels_model",
    },
    "clin_ct_organs": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/MOOSEv2_bspline_organs23062023.zip",
        "filename": "Dataset123_Organs.zip",
        "directory": "Dataset123_Organs",
    },
    "clin_pt_fdg_tumor": {
        "url": "https://example.com/fdg_tumor_model.zip",
        "filename": "clin_pt_fdg_tumor_model.zip",
        "directory": "clin_pt_fdg_tumor_model",
    },
    "clin_ct_all": {
        "url": "https://example.com/ct_all_model.zip",
        "filename": "clin_ct_all_model.zip",
        "directory": "clin_ct_all_model",
    },
    "clin_fdg_pt_ct_all": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/MOOSE-files-24062022.zip",
        "filename": "clin_fdg_pt_ct_all_model.zip",
        "directory": "clin_fdg_pt_ct_all_model",
    },
    "preclin_mr_all": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/preclin_mr_14062023.zip",
        "filename": "Dataset234_Preclin.zip",
        "directory": "Dataset234_Preclin",
    },

}


def check_cuda() -> str:
    """
    This function checks if CUDA is available on the device and prints the device name and number of CUDA devices
    available on the device.

    Returns:
        str: The device to run predictions on, either "cpu" or "cuda".
    """
    if not torch.cuda.is_available():
        print(f"{constants.ANSI_ORANGE}CUDA not available on this device. Predictions will be run on CPU.{constants.ANSI_RESET}")
        return "cpu"
    else:
        device_count = torch.cuda.device_count()
        print(f"{constants.ANSI_GREEN} CUDA is available on this device with {device_count} GPU(s). Predictions will be run on GPU.{constants.ANSI_RESET}")
        return "cuda"
