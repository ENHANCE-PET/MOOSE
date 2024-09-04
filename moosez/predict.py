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
import dask.array as da
import dask
import sys
import torch
from typing import Tuple, List, Any
import SimpleITK
import nibabel as nib
import numpy as np
from halo import Halo
from moosez import constants
from moosez import file_utilities
from moosez import image_processing
from moosez.image_processing import ImageResampler
from moosez.image_processing import NiftiPreprocessor
from moosez.resources import MODELS, map_model_name_to_task_number, check_device
from mpire import WorkerPool

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


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
    task_number = map_model_name_to_task_number(model_name)
    # set the environment variables
    os.environ["nnUNet_results"] = constants.NNUNET_RESULTS_FOLDER

    # Preprocess the image
    temp_input_dir, resampled_image, moose_image_object = preprocess(input_dir, model_name)
    resampled_image_shape = resampled_image.shape
    resampled_image_affine = resampled_image.affine

    # choose the appropriate trainer for the model
    trainer = MODELS[model_name]["trainer"]
    configuration = MODELS[model_name]["configuration"]

    # Construct the command
    command = f'nnUNetv2_predict -i {temp_input_dir} -o {output_dir} -d {task_number} -c {configuration}' \
              f' -f all -tr {trainer} --disable_tta -device {accelerator}'

    # Run the command
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, env=os.environ, stderr=subprocess.DEVNULL)

    original_image_files = file_utilities.get_files(input_dir, '.nii.gz')

    # Postprocess the label
    # if temp_input_dir has more than one nifti file, run logic below

    if len(file_utilities.get_files(output_dir, '.nii.gz')) > 1:
        merge_image_parts(output_dir, resampled_image_shape, resampled_image_affine)
    postprocess(original_image_files[0], output_dir, model_name)

    shutil.rmtree(temp_input_dir)


def preprocess(original_image_directory: str, model_name: str) -> Tuple[str, nib.Nifti1Image, Any]:
    """
    Preprocesses the original images.

    :param original_image_directory: The directory containing the original images.
    :type original_image_directory: str
    :param model_name: The name of the model.
    :type model_name: str
    :return: A tuple containing the path to the temp folder, the resampled image, and the moose_image_object.
    :rtype: Tuple[str, nib.Nifti1Image, Any]
    """
    # create the temp directory
    temp_folder = os.path.join(original_image_directory, constants.TEMP_FOLDER)
    os.makedirs(temp_folder, exist_ok=True)
    original_image_files = file_utilities.get_files(original_image_directory, '.nii.gz')
    org_image = nib.load(original_image_files[0])
    moose_image_object = NiftiPreprocessor(org_image)

    # choose the target spacing for the model to enable preprocessing
    desired_spacing = MODELS[model_name]["voxel_spacing"]

    resampled_image = ImageResampler.resample_image(moose_img_object=moose_image_object,
                                                    interpolation=constants.INTERPOLATION,
                                                    desired_spacing=desired_spacing)
    # if model name has body in it, run logic below
    if "body" in model_name:
        image_processing.write_image(resampled_image, os.path.join(temp_folder, constants.RESAMPLED_IMAGE_FILE_NAME),
                                     False, False)
    else:
        image_processing.write_image(resampled_image, os.path.join(temp_folder, constants.RESAMPLED_IMAGE_FILE_NAME),
                                     moose_image_object.is_large, False)
    return temp_folder, resampled_image, moose_image_object


def postprocess(original_image: str, output_dir: str, model_name: str) -> None:
    """
    Postprocesses the predicted images.

    :param original_image: The path to the original image.
    :type original_image: str
    :param output_dir: The output directory containing the label image.
    :type output_dir: str
    :param model_name: The name of the model.
    :type model_name: str
    :return: None
    :rtype: None
    """
    # [1] Resample the predicted image to the original image's voxel spacing
    predicted_image = file_utilities.get_files(output_dir, '.nii.gz')[0]
    original_header = nib.load(original_image).header
    native_spacing = original_header.get_zooms()
    native_size = original_header.get_data_shape()
    resampled_prediction = ImageResampler.resample_segmentations(input_image_path=predicted_image,
                                                                 desired_spacing=native_spacing,
                                                                 desired_size=native_size)
    multilabel_image = os.path.join(output_dir, MODELS[model_name]["multilabel_prefix"] +
                                    os.path.basename(original_image))
    # if model_name has body in it, run logic below
    if "body" in model_name:
        resampled_prediction_data = resampled_prediction.get_fdata()
        resampled_prediction_new = nib.Nifti1Image(resampled_prediction_data, resampled_prediction.affine,
                                                   resampled_prediction.header)
        image_processing.write_image(resampled_prediction_new, multilabel_image, False, True)
    else:
        image_processing.write_image(resampled_prediction, multilabel_image, False, True)
    os.remove(predicted_image)


def count_output_files(output_dir: str) -> int:
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


def split_and_save(shared_image_data: Tuple[np.ndarray, np.ndarray], z_index: List[Tuple[int, int]],
                   image_chunk_path: str) -> None:
    """
    Split the image and save each part.

    :param shared_image_data: A tuple containing the voxel data of the image and the affine transformation associated with the data.
    :type shared_image_data: Tuple[np.ndarray, np.ndarray]
    :param z_index: A list of tuples containing start and end indices for z-axis split.
    :type z_index: List[Tuple[int, int]]
    :param image_chunk_path: The path to save the image part.
    :type image_chunk_path: str
    :return: None
    :rtype: None
    """
    image_data, image_affine = shared_image_data
    image_part = nib.Nifti1Image(image_data[:, :, z_index[0]:z_index[1]], image_affine)
    nib.save(image_part, image_chunk_path)


def handle_large_image(image: nib.Nifti1Image, save_dir: str) -> List[str]:
    """
    Split a large image into parts and save them.

    :param image: The NIBABEL image.
    :type image: nib.Nifti1Image
    :param save_dir: Directory to save the image parts.
    :type save_dir: str
    :return: List of paths of the original or split image parts.
    :rtype: List[str]
    """

    image_shape = image.shape
    image_data = image.get_fdata()
    image_affine = image.affine

    if np.prod(image_shape) > constants.MATRIX_THRESHOLD and image_shape[2] > constants.Z_AXIS_THRESHOLD:

        # Calculate indices for z-axis split
        z_part = image_shape[2] // 3
        z_indices = [(0, z_part + constants.MARGIN_PADDING),
                     (z_part + 1 - constants.MARGIN_PADDING, z_part * 2 + constants.MARGIN_PADDING),
                     (z_part * 2 + 1 - constants.MARGIN_PADDING, None)]
        filenames = ["subpart01_0000.nii.gz", "subpart02_0000.nii.gz", "subpart03_0000.nii.gz"]
        resampled_chunks_paths = [os.path.join(save_dir, filename) for filename in filenames]

        chunk_data = []
        for z_index, resampled_chunk_path in zip(z_indices, resampled_chunks_paths):
            chunk_data.append({"z_index": z_index,
                               "image_chunk_path": resampled_chunk_path})

        shared_objects = (image_data, image_affine)

        with WorkerPool(n_jobs=3, shared_objects=shared_objects) as pool:
            pool.map(split_and_save, chunk_data)

        return resampled_chunks_paths

    else:
        resampled_image_path = os.path.join(save_dir, constants.RESAMPLED_IMAGE_FILE_NAME)
        nib.save(image, resampled_image_path)

        return [resampled_image_path]


def merge_image_parts(save_dir: str, original_image_shape: Tuple[int, int, int],
                      original_image_affine: np.ndarray) -> str:
    """
    Combine the split image parts back into a single image.

    :param save_dir: Directory where the image parts are saved.
    :type save_dir: str
    :param original_image_shape: The shape of the original image.
    :type original_image_shape: Tuple[int, int, int]
    :param original_image_affine: The affine transformation of the original image.
    :type original_image_affine: np.ndarray
    :return: Path to the merged image.
    :rtype: str
    """
    # Create an empty array with the original image's shape
    merged_image_data = np.zeros(original_image_shape, dtype=np.uint8)

    # Calculate the split index along the z-axis
    z_split_index = original_image_shape[2] // 3

    # Load each part, extract its data, and place it in the correct position in the merged image

    predicted_chunk_filenames = [s.replace("_0000", "") for s in constants.CHUNK_FILENAMES]

    merged_image_data[:, :, :z_split_index] = nib.load(
        os.path.join(save_dir, predicted_chunk_filenames[0])).get_fdata()[:,
                                              :, :-constants.MARGIN_PADDING]
    merged_image_data[:, :, z_split_index:z_split_index * 2] = nib.load(
        os.path.join(save_dir, predicted_chunk_filenames[1])).get_fdata()[:, :,
                                                               constants.MARGIN_PADDING - 1:-constants.MARGIN_PADDING]
    merged_image_data[:, :, z_split_index * 2:] = nib.load(
        os.path.join(save_dir, predicted_chunk_filenames[2])).get_fdata()[
                                                  :, :, constants.MARGIN_PADDING - 1:]

    # Create a new Nifti1Image with the merged data and the original image's affine transformation
    merged_image = nib.Nifti1Image(merged_image_data, original_image_affine)

    # remove the split image parts
    files_to_remove = glob.glob(os.path.join(save_dir, constants.CHUNK_PREFIX + "*"))
    for file in files_to_remove:
        os.remove(file)

    # write the merged image to disk
    merged_image_path = os.path.join(save_dir, constants.RESAMPLED_IMAGE_FILE_NAME)
    nib.save(merged_image, merged_image_path)

    return merged_image_path


def initialize_predictor(model_name: str, accelerator: str) -> nnUNetPredictor:
    """
    Initializes the model for prediction.

    :param model_name: The name of the model.
    :type model_name: str
    :param accelerator: The accelerator for prediction.
    :type model_name: str
    :return: The initialized predictor object.
    :rtype: nnUNetPredictor
    """
    model_folder_name = MODELS[model_name]["directory"]
    trainer = MODELS[model_name]["trainer"]
    configuration = MODELS[model_name]["configuration"]
    planner = MODELS[model_name]["planner"]
    device = torch.device(accelerator)
    predictor = nnUNetPredictor(allow_tqdm=False, device=device)
    predictor.initialize_from_trained_model_folder(os.path.join(constants.NNUNET_RESULTS_FOLDER, model_folder_name, f"{trainer}__{planner}__{configuration}"), use_folds=("all"))
    return predictor


@dask.delayed
def process_case(preprocessor, chunk: np.ndarray, chunk_properties: dict, predictor, location: tuple):
    data, seg = preprocessor.run_case_npy(chunk,
                                          None,
                                          chunk_properties,
                                          predictor.plans_manager,
                                          predictor.configuration_manager,
                                          predictor.dataset_json)
    return {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': chunk_properties, 'ofile': None,
            'location': location}


def preprocessing_iterator_from_dask_array(dask_array: da.Array, image_properties: dict, predictor):
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)

    chunk_indices = []
    delayed_chunks = dask_array.to_delayed()
    delayed_tasks = []

    for chunk_index in np.ndindex(delayed_chunks.shape):
        chunk_indices.append(chunk_index)
        image_chunk = delayed_chunks[chunk_index]
        delayed_task = dask.delayed(process_case)(preprocessor, image_chunk, image_properties, predictor, chunk_index)
        delayed_tasks.append(delayed_task)

    results = dask.compute(*delayed_tasks)
    iterator = iter(results)

    return iterator, chunk_indices


def reconstruct_array_from_chunks(chunks: list[np.array], chunk_positions: list[tuple], original_shape: tuple):
    reconstructed_array = np.empty(original_shape, dtype=chunks[0].dtype)

    for chunk, position in zip(chunks, chunk_positions):
        chunk = chunk[None, ...]
        slices = tuple(slice(pos * csize, pos * csize + csize) for pos, csize in zip(position, chunk.shape))
        reconstructed_array[slices] = chunk

    return np.squeeze(reconstructed_array)


def predict_from_array_by_iterator(image_array: np.ndarray, model_name: str, accelerator: str = None):
    image_array = image_array[None, ...]
    chunks = [axis / image_processing.ImageResampler.chunk_along_axis(axis) for axis in image_array.shape]
    prediction_array = da.from_array(image_array, chunks=chunks)

    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            if accelerator is None:
                accelerator = check_device()
            predictor = initialize_predictor(model_name, accelerator)
            image_properties = {
                'spacing': MODELS[model_name]["voxel_spacing"]
            }
            iterator, chunk_locations = preprocessing_iterator_from_dask_array(prediction_array, image_properties,
                                                                               predictor)
            segmentations = predictor.predict_from_data_iterator(iterator)
            combined_segmentations = reconstruct_array_from_chunks(segmentations, chunk_locations, prediction_array.shape)
            return combined_segmentations

        finally:
            sys.stdout = old_stdout


def prediction_pipeline(image, model_name, accelerator):

    desired_spacing = MODELS[model_name]["voxel_spacing"]
    resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline',
                                                                                          desired_spacing)

    segmentation_array = predict_from_array_by_iterator(resampled_array, model_name, accelerator)
    return segmentation_array

def limit_fov(image, model_to_limit_fov_from, fov_label, accelerator):

    segmentation_array = prediction_pipeline(image, model_to_limit_fov_from, accelerator)

    # Convert the images to numpy arrays
    image_array = SimpleITK.GetArrayFromImage(image)

    # Find the z-axis indices where the label matches the target intensity
    z_indices = np.where(segmentation_array == fov_label)[
        0]  # Note: SimpleITK uses [z,y,x] indexing
    z_min, z_max = np.min(z_indices), np.max(z_indices)

    # Crop the CT data along the z-axis
    limited_fov_array = image_array[z_min:z_max + 1, :, :]  # Note: SimpleITK uses [z,y,x] indexing

    # Convert the cropped numpy array back to a SimpleITK image
    limited_fov_image = SimpleITK.GetImageFromArray(limited_fov_array)

    # Set the same origin, spacing, and direction as the original image
    limited_fov_image.SetOrigin(image.GetOrigin())
    limited_fov_image.SetSpacing(image.GetSpacing())
    limited_fov_image.SetDirection(image.GetDirection())

    return limited_fov_image, {"z_min": z_min, "z_max": z_max, "original_shape": image_array.shape, "original_image": image}