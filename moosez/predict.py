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

import os
import shutil
import subprocess

from halo import Halo

from moosez import constants
from moosez import file_utilities
from moosez import image_processing


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
        return 205
    elif model_name == "clin_ct_fat":
        return 206
    elif model_name == "clin_ct_vessels":
        return 207
    elif model_name == "clin_ct_organs":
        return 123
    elif model_name == "clin_pt_fdg_tumor":
        return 209
    elif model_name == "clin_ct_all":
        return 210
    elif model_name == "clin_fdg_pt_ct_all":
        return 211
    elif model_name == "preclin_mr_all":
        return 212
    else:
        raise Exception(f"Error: The model name '{model_name}' is not valid.")


def predict(model_name: str, input_dir: str, output_dir: str, accelerator: str):
    """
    Runs the prediction using nnunet_predict.
    :param model_name: The name of the model.
    :param input_dir: The input directory.
    :param output_dir: The output directory.
    :param accelerator: The accelerator to use.
    :return: None
    """
    task_number = map_model_name_to_task_number(model_name)
    # set the environment variables
    os.environ["nnUNet_results"] = constants.NNUNET_RESULTS_FOLDER

    # Preprocess the image
    temp_input_dir, resampled_image = preprocess(input_dir)
    subprocess.run(f'nnUNetv2_predict -i {temp_input_dir} -o {output_dir} -d {task_number} -c 3d_fullres -f all'
                   f' -tr nnUNetTrainer_2000epochs_NoMirroring'
                   f' --disable_tta -device {accelerator}',
                   shell=True, stdout=subprocess.DEVNULL, env=os.environ)
    original_image_files = file_utilities.get_files(input_dir, '.nii.gz')

    # Postprocess the label
    postprocess(original_image_files[0], output_dir)
    shutil.rmtree(temp_input_dir)


def preprocess(original_image_directory: str):
    """
    Preprocesses the original images.
    :param original_image_directory: The directory containing the original images.
    :return: temp_folder: The path to the temp folder.
    """
    # create the temp directory
    temp_folder = os.path.join(original_image_directory, constants.TEMP_FOLDER)
    os.makedirs(temp_folder, exist_ok=True)
    original_image_files = file_utilities.get_files(original_image_directory, '.nii.gz')
    resampled_image = os.path.join(temp_folder, 'resampled_image_0000.nii.gz')

    # [1] Resample the images to 1.5 mm isotropic voxel size
    resampled_image = image_processing.resample(input_image_path=original_image_files[0],
                                                output_image_path=resampled_image,
                                                interpolation='linear',
                                                desired_spacing=constants.VOXEL_SPACING)
    return temp_folder, resampled_image


def postprocess(original_image, output_dir):
    """
    Postprocesses the predicted images.
    :param original_image: The original image.
    :param output_dir: The output directory containing the label image.
    :return: None
    """
    # [1] Resample the predicted image to the original image's voxel spacing
    predicted_image = file_utilities.get_files(output_dir, '.nii.gz')[0]
    multilabel_image = os.path.join(output_dir, constants.MULTILABEL_SUFFIX + os.path.basename(original_image))
    image_processing.resample(input_image_path=predicted_image,
                              output_image_path=multilabel_image,
                              interpolation='nearest',
                              desired_spacing=image_processing.get_spacing(original_image))
    os.remove(predicted_image)


def count_output_files(output_dir):
    """
    Counts the number of files in the specified output directory.
    Parameters:
        output_dir (str): The path to the output directory.
    Returns:
        The number of files in the output directory.
    """
    return len([name for name in os.listdir(output_dir) if
                os.path.isfile(os.path.join(output_dir, name)) and name.endswith('.nii.gz')])


def monitor_output_directory(output_dir, total_files, spinner):
    """
    Continuously monitors the specified output directory for new files and updates the progress bar accordingly.
    Parameters:
        output_dir (str): The path to the output directory.
        total_files (int): The total number of files that are expected to be generated in the output directory.
        spinner (Halo): The spinner that displays the progress of the segmentation process.
    Returns:
        None
    """
    files_processed = count_output_files(output_dir)
    while files_processed < total_files:
        new_files_processed = count_output_files(output_dir)
        if new_files_processed > files_processed:
            spinner.text = f'Processed {new_files_processed} of {total_files} files'
            spinner.spinner = 'dots'
        files_processed = new_files_processed
