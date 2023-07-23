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
import logging
from moosez import constants

MODELS = {
    "clin_ct_bones": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/Task201_CT_Bones.zip",
        "filename": "Task201_CT_Bones.zip",
        "directory": "Task201_CT_Bones",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Bones_"
    },
    "clin_ct_ribs": {
        "url": "https://example.com/bones_model.zip",
        "filename": "clin_ct_ribs_model.zip",
        "directory": "clin_ct_ribs_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Ribs_"

    },
    "clin_ct_vertebrae": {
        "url": "https://example.com/vertebrae_model.zip",
        "filename": "clin_ct_vertebrae_model.zip",
        "directory": "clin_ct_vertebrae_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_VertebralBody_"
    },
    "clin_ct_muscles": {
        "url": "https://example.com/muscles_model.zip",
        "filename": "clin_ct_muscles_model.zip",
        "directory": "clin_ct_muscles_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Muscles_"
    },
    "clin_ct_lungs": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/clin_ct_lungs_24062023.zip",
        "filename": "Dataset333_HMS3dlungs.zip",
        "directory": "Dataset333_HMS3dlungs",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Lungs_"
    },
    "clin_ct_fat": {
        "url": "https://example.com/fat_model.zip",
        "filename": "clin_ct_fat_model.zip",
        "directory": "clin_ct_fat_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Fat_"
    },
    "clin_ct_vessels": {
        "url": "https://example.com/vessels_model.zip",
        "filename": "clin_ct_vessels_model.zip",
        "directory": "clin_ct_vessels_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Vessels_"
    },
    "clin_ct_organs": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/MOOSEv2_bspline_organs23062023.zip",
        "filename": "Dataset123_Organs.zip",
        "directory": "Dataset123_Organs",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Organs_"
    },
    "clin_pt_fdg_tumor": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/Dataset789_Tumors.zip",
        "filename": "Dataset789_Tumors.zip",
        "directory": "Dataset789_Tumors",
        "trainer": "nnUNetTrainerDA5",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "PT_FDG_Tumor_"
    },
    "clin_ct_all": {
        "url": "https://example.com/ct_all_model.zip",
        "filename": "clin_ct_all_model.zip",
        "directory": "clin_ct_all_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_all_"
    },
    "clin_fdg_pt_ct_all": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/MOOSE-files-24062022.zip",
        "filename": "clin_fdg_pt_ct_all_model.zip",
        "directory": "clin_fdg_pt_ct_all_model",
        "trainer": "nnUNetTrainer_2000epochs_NoMirroring",
        "voxel_spacing": [2.0, 2.0, 2.0],
        "multilabel_prefix": "CT_fdg_pt_all_"
    },
    "preclin_mr_all": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/preclin_mr_14062023.zip",
        "filename": "Dataset234_Preclin.zip",
        "directory": "Dataset234_Preclin",
        "trainer": "nnUNetTrainerNoMirroring",
        "voxel_spacing": [0.15, 0.15, 0.15],
        "multilabel_prefix": "Preclin_MR_all_"
    },
    "clin_ct_body": {
        "url": "https://moose-files.s3.eu-de.cloud-object-storage.appdomain.cloud/Dataset696_BodyContour.zip",
        "filename": "Dataset696_BodyContour.zip",
        "directory": "Dataset696_BodyContour",
        "trainer": "nnUNetTrainer",
        "voxel_spacing": [1.5, 1.5, 1.5],
        "multilabel_prefix": "CT_Body_"
    },
}

AVAILABLE_MODELS = ["clin_ct_bones",
                    "clin_ct_ribs",
                    "clin_ct_vertebrae",
                    "clin_ct_muscles",
                    "clin_ct_lungs",
                    "clin_ct_fat",
                    "clin_ct_vessels",
                    "clin_ct_organs",
                    "clin_pt_fdg_tumor",
                    "clin_ct_all",
                    "clin_fdg_pt_ct_all",
                    "clin_ct_body",
                    "preclin_mr_all"]


def expected_modality(model_name: str) -> dict:
    """
    Display expected modality for the model.
    :param model_name: The name of the model.
    :return: The expected modality for the model.
    """
    models = {
        "clin_ct_bones": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Bones"},
        "clin_ct_ribs": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Ribs"},
        "clin_ct_vertebrae": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Vertebral bodies"},
        "clin_ct_muscles": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Muscles"},
        "clin_ct_lungs": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Lungs"},
        "clin_ct_fat": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Fat"},
        "clin_ct_vessels": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Vessels"},
        "clin_ct_organs": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Organs"},
        "clin_pt_fdg_tumor": {"Imaging": "Clinical", "Modality": "PET", "Tissue of interest": "Tumor"},
        "clin_ct_all": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "All regions"},
        "clin_fdg_pt_ct_all": {"Imaging": "Clinical", "Modality": "FDG-PET-CT", "Tissue of interest": "All regions"},
        "clin_ct_body": {"Imaging": "Clinical", "Modality": "CT", "Tissue of interest": "Body"},
        "preclin_mr_all": {"Imaging": "Pre-clinical", "Modality": "MR", "Tissue of interest": "All regions"},
    }

    if model_name in models:
        model = models[model_name]
        model["Model name"] = model_name
        return model

    logging.error(" Requested model is not available. Please check the model name.")
    return {"Error": "Requested model is not available. Please check the model name."}


def map_model_name_to_task_number(model_name: str):
    """
    Maps the model name to the task number.
    :param model_name: The name of the model.
    :return: The task number.
    """
    if model_name == "clin_ct_bones":
        return 201
    elif model_name == "clin_ct_ribs":
        return 202
    elif model_name == "clin_ct_vertebrae":
        return 203
    elif model_name == "clin_ct_muscles":
        return 204
    elif model_name == "clin_ct_lungs":
        return 333
    elif model_name == "clin_ct_fat":
        return 206
    elif model_name == "clin_ct_vessels":
        return 207
    elif model_name == "clin_ct_organs":
        return 123
    elif model_name == "clin_pt_fdg_tumor":
        return 789
    elif model_name == "clin_ct_all":
        return 210
    elif model_name == "clin_fdg_pt_ct_all":
        return 211
    elif model_name == "preclin_mr_all":
        return 234
    elif model_name == "clin_ct_body":
        return 696
    else:
        raise Exception(f"Error: The model name '{model_name}' is not valid.")


def check_cuda() -> str:
    """
    This function checks if CUDA is available on the device and prints the device name and number of CUDA devices
    available on the device.

    Returns:
        str: The device to run predictions on, either "cpu" or "cuda".
    """
    if not torch.cuda.is_available():
        print(
            f"{constants.ANSI_ORANGE}CUDA not available on this device. Predictions will be run on CPU.{constants.ANSI_RESET}")
        return "cpu"
    else:
        device_count = torch.cuda.device_count()
        print(
            f"{constants.ANSI_GREEN} CUDA is available on this device with {device_count} GPU(s). Predictions will be run on GPU.{constants.ANSI_RESET}")
        return "cuda"
