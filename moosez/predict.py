#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
#         Manuel Pires
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module contains the necessary functions for prediction using the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform prediction.
#
# ----------------------------------------------------------------------------------------------------------------------

import glob
import os
import shutil
import subprocess
from typing import Tuple, List, Any

import nibabel as nib
import numpy as np
from halo import Halo
from moosez import constants
from moosez import file_utilities
from moosez import image_processing
from moosez.image_processing import ImageResampler
from moosez.image_processing import NiftiPreprocessor
from moosez.resources import MODELS, map_model_name_to_task_number
from nnunetv2.paths import nnUNet_results
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from mpire import WorkerPool


def initialize_model(model_name: str) -> nnUNetPredictor:
    """
    Initializes the model for prediction.

    :param model_name: The name of the model.
    :type model_name: str
    :return: The initialized predictor object.
    :rtype: nnUNetPredictor
    """
    model_folder_name = MODELS[model_name]["directory"]
    trainer = MODELS[model_name]["trainer"]
    configuration = MODELS[model_name]["configuration"]
    planner = MODELS[model_name]["planner"]
    predictor = nnUNetPredictor(allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(os.path.join(constants.NNUNET_RESULTS_FOLDER, model_folder_name,
                                                                f"{trainer}__{planner}__{configuration}"), use_folds=("all"))
    return predictor


def prediction_pipeline(predictor: nnUNetPredictor, input_dir: str, output_dir: str, model_name: str) -> None:
    """
    Uses the initialized model to infer the input image.

    :param predictor: The predictor object.
    :type predictor: nnUNetPredictor
    :param input_dir: The input directory.
    :type input_dir: str
    :param output_dir: The output directory.
    :type output_dir: str
    :param model_name: The name of the model.
    :type model_name: str
    :return: None
    :rtype: None
    """
    image, nnunet_dict, properties_dict = preprocess(input_dir, model_name)
    segmentation = predictor.predict_from_list_of_npy_arrays(image, None, nnunet_dict, None) # Returns a np.array

    if len(segmentation) > 1:
        predicted_image = merge_image_parts(segmentation, properties_dict["resampled_shape"], properties_dict["resampled_affine"])
    else:
        predicted_image = nib.Nifti1Image(segmentation[0], nnunet_dict["nibabel_stuff"]["original_affine"],
                                          properties_dict["resampled_header"])

    postprocess(predicted_image, input_dir, output_dir, model_name, properties_dict["original_header"])



def preprocess(original_image_directory: str, model_name: str) -> Tuple[np.array, dict, dict]:
    """
    Preprocesses the original images.

    :param original_image_directory: The directory containing the original images.
    :type original_image_directory: str
    :param model_name: The name of the model.
    :type model_name: str
    :return: A tuple containing the array with the image data, a dictionary for the predictor, and a dictionary for postprocessing.
    :rtype: Tuple[np.array, dict, dict]
    """

    prefix = MODELS[model_name]["multilabel_prefix"].split("_")[1]
    original_image_files = file_utilities.get_files(original_image_directory, prefix, ('.nii.gz', '.nii'))
    org_image = nib.load(original_image_files[0])
    original_header = org_image.header
    moose_image_object = NiftiPreprocessor(org_image)

    # choose the target spacing for the model to enable preprocessing
    desired_spacing = MODELS[model_name]["voxel_spacing"]

    resampled_image = ImageResampler.resample_image(moose_img_object=moose_image_object,
                                                    interpolation=constants.INTERPOLATION,
                                                    desired_spacing=desired_spacing)
    properties_dict = {"original_header": original_header, "resampled_header": resampled_image.header,
                       "resampled_shape": resampled_image.shape, "resampled_affine": resampled_image.affine}

    # if model name has body in it, run logic below
    if "body" in model_name:
        image_data = resampled_image.get_fdata().transpose((2, 1, 0))[None] # According to nnUNet the transpose is to make it consistent with stik loading
        nnunet_dict = {
            "nibabel_stuff": {
                "original_affine": resampled_image.affine
            },
            "spacing": [float(i) for i in resampled_image.header.get_zooms()[::-1]] # Also retrieved like this to make it consistent with sitk
        }

    else:
        image_data, nnunet_dict = image_processing.chunk_image(resampled_image, moose_image_object.is_large)

    return image_data, nnunet_dict, properties_dict


def postprocess(predicted_image, original_image: str, output_dir: str, model_name: str, original_header:nib.Nifti1Header) -> None:
    """
    Postprocesses the predicted images.

    :param predicted_image: Image infered by nnUNet.
    :type predicted_image: str
    :param original_image: The path to the original image.
    :type original_image: str
    :param output_dir: The output directory containing the label image.
    :type output_dir: str
    :param model_name: The name of the model.
    :type model_name: str
    :param original_header: Header of the image infered.
    :type original_header: nib.Nifti1Header
    :return: None
    :rtype: None
    """
    # [1] Resample the predicted image to the original image's voxel spacing
    native_spacing = original_header.get_zooms()
    native_size = original_header.get_data_shape()
    resampled_prediction = ImageResampler.resample_segmentations(input_image=predicted_image,
                                                                 desired_spacing=native_spacing,
                                                                 desired_size=native_size)
    multilabel_image = os.path.join(output_dir, MODELS[model_name]["multilabel_prefix"] +
                                    os.path.basename(original_image))
    image_processing.write_segmentation(resampled_prediction, multilabel_image)


def merge_image_parts(segmentations: list, original_image_shape: Tuple[int, int, int],
                      original_image_affine: np.ndarray) -> nib.Nifti1Image:
    """
    Combine the split image parts back into a single image.

    :param segmentations: List with predicted segmentations for each chunk.
    :type segmentations: list
    :param original_image_shape: The shape of the original image.
    :type original_image_shape: Tuple[int, int, int]
    :param original_image_affine: The affine transformation of the original image.
    :type original_image_affine: np.ndarray
    :return: Merged segmentation image.
    :rtype: nib.Nifti1Image
    """
    # Create an empty array with the original image's shape
    merged_image_data = np.zeros(original_image_shape, dtype=np.uint8)
    # Calculate the split index along the z-axis
    z_split_index = original_image_shape[2] // 3


    merged_image_data[:, :, :z_split_index] = segmentations[0].transpose((2,1,0))[:,:, :-constants.MARGIN_PADDING]
    merged_image_data[:, :, z_split_index:z_split_index * 2] = segmentations[1].transpose((2,1,0))[:, :, constants.MARGIN_PADDING - 1:-constants.MARGIN_PADDING]
    merged_image_data[:, :, z_split_index * 2:] = segmentations[2].transpose((2,1,0))[:, :, constants.MARGIN_PADDING - 1:]

    # Create a new Nifti1Image with the merged data and the original image's affine transformation
    merged_image = nib.Nifti1Image(merged_image_data.transpose((2,1,0)), original_image_affine)


    return merged_image
