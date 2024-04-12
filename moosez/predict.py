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
from typing import Tuple, List, Any, Sequence

import torch
import nibabel as nib
import numpy as np
from halo import Halo
from mpire import WorkerPool

from moosez import constants, file_utilities, image_processing
from moosez.image_processing import ImageResampler, NiftiPreprocessor, get_pixdim_from_affine
from moosez.resources import MODELS, map_model_name_to_task_number
from moosez.benchmarking.profiler import Profiler


def predict(model_name: str, input_dir: str, output_dir: str, accelerator: str) -> None:
    """
    Runs the prediction using nnunet_predict.

    :param model_name: The name of the model.
    :type model_name: str
    :param input_dir: The input directory.
    :type input_dir: str
    :param output_dir: The output directory.
    :type output_dir: str
    :param accelerator: The accelerator to use.
    :type accelerator: str
    :return: None
    :rtype: None
    """
    profiler = Profiler()
    profiler.set_section("preprocessing")
    task_number = map_model_name_to_task_number(model_name)
    # set the environment variables
    os.environ["nnUNet_results"] = constants.NNUNET_RESULTS_FOLDER

    # Preprocess the image
    voxel_spacing = MODELS[model_name]['voxel_spacing']
    temp_input_dir, input_resampled_shape, resampled_image_affine, resampled_image = preprocess_monai(input_dir, voxel_spacing, accelerator)

    # choose the appropriate trainer for the model
    trainer = MODELS[model_name]["trainer"]
    configuration = MODELS[model_name]["configuration"]

    # Construct the command
    command = f'nnUNetv2_predict -i {temp_input_dir} -o {output_dir} -d {task_number} -c {configuration}' \
              f' -f all -tr {trainer} --disable_tta -device {accelerator}'

    # Run the command
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, env=os.environ, stderr=subprocess.DEVNULL)

    # Postprocess the label
    # if temp_input_dir has more than one nifti file, run logic below
    profiler.set_section("post-processing")
    original_spacing = get_pixdim_from_affine(resampled_image.meta['original_affine'])
    # takes last label id to define number of classes for one-hot encoding
    n_class = sorted(constants.ORGAN_INDICES['clin_ct_organs'].keys())[-1] + 1
    label_prefix = MODELS[model_name]['multilabel_prefix']
    postprocess_monai(output_dir, original_spacing, input_resampled_shape, label_prefix, n_class, accelerator)

    shutil.rmtree(temp_input_dir)


def preprocess_monai(
        original_image_directory: str, voxel_spacing: Sequence[float], accelerator: str):
    """
    Preprocesses the original images using monai.

    Args:
        original_image_directory: The directory containing the original images.
        voxel_spacing: The target voxel spacing to use.
        accelerator: Specify the target device.

    Returns:
        temp_folder: The path to the temp folder.
        resampled_image: The resampled image of `monai.data.MetaTensor` type.
    """
    temp_folder = os.path.join(original_image_directory, constants.TEMP_FOLDER)
    os.makedirs(temp_folder, exist_ok=True)
    original_image_files = file_utilities.get_files(original_image_directory, ".nii.gz")

    resample_transform = ImageResampler.monai_resampling(
        interpolation=constants.INTERPOLATION,
        desired_spacing=voxel_spacing,
        device=accelerator,
        output_dir=temp_folder)
    resampled_image = resample_transform(original_image_files[0])
    resampled_shape = tuple(resampled_image.squeeze().shape)
    resampled_affine = np.copy(resampled_image.affine)

    # Release cuda memory because garbage collector not called after Transform
    torch.cuda.empty_cache()

    return temp_folder, resampled_shape, resampled_affine, resampled_image

def postprocess_monai(
        output_dir: str,
        original_spacing: Sequence[int],
        input_resampled_shape: Sequence[int],
        label_prefix: str,
        n_class: int,
        accelerator: str):
    """
    Postprocesses the predicted segmentation using monai.

    Args:
        output_dir: The directory containing the resulting segmentation.
        original_spacing: The original image spacing to resample to.
        input_resampled_shape: The shape of the resampled input image before chunk splitter.
        label_prefix: Prefix to be appended to the filenames.
        n_class: The number of classes for the Merger.
        accelerator: Specify the target device.
    """

    predicted_images = file_utilities.get_files(output_dir, '.nii.gz')
    seg_resample_transform = ImageResampler.monai_segmentation_resampling(
        interpolation=constants.INTERPOLATION,
        original_spacing=original_spacing,
        input_resampled_shape=input_resampled_shape,
        label_prefix=label_prefix,
        n_class=n_class,
        device=accelerator)
    seg_resample_transform(predicted_images)


def count_output_files(output_dir):
    """
    Counts the number of files in the specified output directory.

    :param output_dir: The path to the output directory.
    :type output_dir: str
    :return: The number of files in the output directory.
    :rtype: int
    """
    return len([name for name in os.listdir(output_dir) if
                os.path.isfile(os.path.join(output_dir, name)) and name.endswith('.nii.gz')])


def monitor_output_directory(output_dir: str, total_files: int, spinner: Halo) -> None:
    """
    Continuously monitors the specified output directory for new files and updates the progress bar accordingly.

    :param output_dir: The path to the output directory.
    :type output_dir: str
    :param total_files: The total number of files that are expected to be generated in the output directory.
    :type total_files: int
    :param spinner: The spinner that displays the progress of the segmentation process.
    :type spinner: Halo
    :return: None
    :rtype: None
    """
    files_processed = count_output_files(output_dir)
    while files_processed < total_files:
        new_files_processed = count_output_files(output_dir)
        if new_files_processed > files_processed:
            spinner.text = f'Processed {new_files_processed} of {total_files} files'
            spinner.spinner = 'dots'
        files_processed = new_files_processed
