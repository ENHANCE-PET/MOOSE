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
import glob
from halo import Halo

from moosez import constants
from moosez import file_utilities
from moosez import image_processing
import numpy as np
import nibabel as nib
from pathlib import Path


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
        return 124
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
        return 234
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
    temp_input_dir, resampled_image = preprocess(input_dir, model_name)
    resampled_image_shape = nib.load(resampled_image).shape
    resampled_image_affine = nib.load(resampled_image).affine
    os.remove(resampled_image)

    # check if model is clinical or precinical

    if model_name.startswith('clin'):
        subprocess.run(f'nnUNetv2_predict -i {temp_input_dir} -o {output_dir} -d {task_number} -c 3d_fullres -f all'
                       f' -tr nnUNetTrainer_2000epochs_NoMirroring'
                       f' --disable_tta -device {accelerator}',
                       shell=True, stdout=subprocess.DEVNULL, env=os.environ)

    else:
        subprocess.run(f'nnUNetv2_predict -i {temp_input_dir} -o {output_dir} -d {task_number} -c 3d_fullres -f all'
                       f' -tr nnUNetTrainerNoMirroring'
                       f' --disable_tta -device {accelerator}',
                       shell=True, stdout=subprocess.DEVNULL, env=os.environ)

    original_image_files = file_utilities.get_files(input_dir, '.nii.gz')

    # Postprocess the label
    merge_image_parts(output_dir, resampled_image_shape, resampled_image_affine)
    postprocess(original_image_files[0], output_dir)

    shutil.rmtree(temp_input_dir)


def preprocess(original_image_directory: str, model_name: str):
    """
    Preprocesses the original images.
    :param original_image_directory: The directory containing the original images.
    :param model_name: The name of the model.
    :return: temp_folder: The path to the temp folder.
    """
    # create the temp directory
    temp_folder = os.path.join(original_image_directory, constants.TEMP_FOLDER)
    os.makedirs(temp_folder, exist_ok=True)
    original_image_files = file_utilities.get_files(original_image_directory, '.nii.gz')
    resampled_image = os.path.join(temp_folder, constants.RESAMPLED_IMAGE_FILE_NAME)

    # check if the model is a clinical model or preclinical model
    if model_name.startswith('clin'):

        # [1] Resample the images to 1.5 mm isotropic voxel size
        resampled_image = image_processing.resample(input_image_path=original_image_files[0],
                                                    output_image_path=resampled_image,
                                                    interpolation='bspline',
                                                    desired_spacing=constants.CLINICAL_VOXEL_SPACING)
    else:
        # [1] Resample the images to 0.5 mm isotropic voxel size
        resampled_image = image_processing.resample(input_image_path=original_image_files[0],
                                                    output_image_path=resampled_image,
                                                    interpolation='bspline',
                                                    desired_spacing=constants.PRECLINICAL_VOXEL_SPACING)
    # [2] Chunk if the image is too large

    handle_large_image(resampled_image, temp_folder)

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


def check_image_size(image_shape):
    """
    Check if the image size exceeds the threshold or the z-axis length is less than the minimum.

    Args:
        image_shape (tuple): The shape of the image.

    Returns:
        bool: True if conditions are met, False otherwise.
    """
    return np.prod(image_shape) > constants.MATRIX_THRESHOLD and image_shape[2] > constants.Z_AXIS_THRESHOLD


def split_and_save(image_data, affine, save_dir, z_indices, filenames):
    """
    Split the image and save each part.

    Args:
        image_data (np.ndarray): The voxel data of the image.
        affine (np.ndarray): The affine transformation associated with the data.
        save_dir: The directory to save the image part.
        z_indices (list): List of tuples containing start and end indices for z-axis split.
        filenames (list): List of filenames to save each image part.
    """
    for z_index, filename in zip(z_indices, filenames):
        image_part = nib.Nifti1Image(image_data[:, :, z_index[0]:z_index[1]], affine)
        nib.save(image_part, os.path.join(save_dir, filename))


def handle_large_image(image_path, save_dir):
    """
    Split a large image into parts and save them.

    Args:
        image_path: path to the image.
        save_dir: Directory to save the image parts.

    Returns:
        list: List of paths of the original or split image parts.
    """
    image = nib.load(image_path)
    image_shape = image.shape
    image_data = image.get_fdata()

    if check_image_size(image_shape):

        # Calculate indices for z-axis split
        z_part = image_shape[2] // 3
        z_indices = [(0, z_part + constants.MARGIN_PADDING),
                     (z_part + 1 - constants.MARGIN_PADDING, z_part * 2 + constants.MARGIN_PADDING),
                     (z_part * 2 + 1 - constants.MARGIN_PADDING, None)]
        filenames = ["subpart01_0000.nii.gz", "subpart02_0000.nii.gz", "subpart03_0000.nii.gz"]

        split_and_save(image_data, image.affine, save_dir, z_indices, filenames)

        return [os.path.join(save_dir, filename) for filename in filenames]

    else:
        return [image_path]


def merge_image_parts(save_dir, original_image_shape, original_image_affine):
    """
    Combine the split image parts back into a single image.

    Args:
        save_dir (str): Directory where the image parts are saved.
        original_image_shape (tuple): The shape of the original image.
        original_image_affine (np.ndarray): The affine transformation of the original image.

    Returns:
        merged_image_path (str): Path to the merged image.
    """
    # Create an empty array with the original image's shape
    merged_image_data = np.zeros(original_image_shape, dtype=np.uint8)

    # Calculate the split index along the z-axis
    z_split_index = original_image_shape[2] // 3

    # Load each part, extract its data, and place it in the correct position in the merged image
    merged_image_data[:, :, :z_split_index] = nib.load(os.path.join(save_dir, "subpart01.nii.gz")).get_fdata()[:,
                                              :, :-constants.MARGIN_PADDING]
    merged_image_data[:, :, z_split_index:z_split_index * 2] = nib.load(
        os.path.join(save_dir, "subpart02.nii.gz")).get_fdata()[:, :,
                                                               constants.MARGIN_PADDING - 1:-constants.MARGIN_PADDING]
    merged_image_data[:, :, z_split_index * 2:] = nib.load(os.path.join(save_dir, "subpart03.nii.gz")).get_fdata()[
                                                  :, :, constants.MARGIN_PADDING - 1:]

    # Create a new Nifti1Image with the merged data and the original image's affine transformation
    merged_image = nib.Nifti1Image(merged_image_data, original_image_affine)

    # remove the split image parts
    files_to_remove = glob.glob(os.path.join(save_dir, "subpart*"))
    for file in files_to_remove:
        os.remove(file)

    # write the merged image to disk
    merged_image_path = os.path.join(save_dir, constants.RESAMPLED_IMAGE_FILE_NAME)
    nib.save(merged_image, merged_image_path)

    return merged_image_path
