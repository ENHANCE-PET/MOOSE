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

import dask
import sys
import torch
import numpy as np
import SimpleITK
from moosez import models
from moosez import image_processing
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def initialize_predictor(model: models.Model, accelerator: str) -> nnUNetPredictor:
    """
    Initializes the model for prediction.

    :param model: The model object.
    :type model: Model
    :param accelerator: The accelerator for prediction.
    :type accelerator: str
    :return: The initialized predictor object.
    :rtype: nnUNetPredictor
    """
    device = torch.device(accelerator)
    predictor = nnUNetPredictor(allow_tqdm=False, device=device)
    predictor.initialize_from_trained_model_folder(model.configuration_directory, use_folds=("all",))
    return predictor


@dask.delayed
def process_case(preprocessor, chunk: np.ndarray, chunk_properties: dict, predictor: nnUNetPredictor, location: tuple) -> dict:
    data, seg = preprocessor.run_case_npy(chunk,
                                          None,
                                          chunk_properties,
                                          predictor.plans_manager,
                                          predictor.configuration_manager,
                                          predictor.dataset_json)

    data_tensor = torch.from_numpy(data).contiguous()
    if predictor.device == "cuda":
        data_tensor = data_tensor.pin_memory()

    return {'data': data_tensor, 'data_properties': chunk_properties, 'ofile': None, 'location': location}


def preprocessing_iterator_from_array(image_array: np.ndarray, image_properties: dict, predictor: nnUNetPredictor) -> (iter, list):
    overlap_per_dimension = (0, 20, 20, 20)
    splits = image_processing.ImageChunker.determine_splits(image_array)
    chunks, locations = image_processing.ImageChunker.array_to_chunks(image_array, splits, overlap_per_dimension)

    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)

    delayed_tasks = []
    for image_chunk, location in zip(chunks, locations):
        delayed_task = dask.delayed(process_case)(preprocessor, image_chunk, image_properties, predictor, location)
        delayed_tasks.append(delayed_task)

    results = dask.compute(*delayed_tasks)
    iterator = iter(results)

    return iterator, locations


def predict_from_array_by_iterator(image_array: np.ndarray, model: models.Model, accelerator: str, nnunet_log_filename: str = None):
    image_array = image_array[None, ...]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    nnunet_log_file = None
    if nnunet_log_filename is not None:
        nnunet_log_file = open(nnunet_log_filename, "a")
        sys.stdout = nnunet_log_file
        sys.stderr = nnunet_log_file

    try:
        predictor = initialize_predictor(model, accelerator)
        image_properties = {
            'spacing': model.voxel_spacing
        }

        iterator, chunk_locations = preprocessing_iterator_from_array(image_array, image_properties, predictor)
        segmentations = predictor.predict_from_data_iterator(iterator)
        segmentations = [segmentation[None, ...] for segmentation in segmentations]
        combined_segmentations = image_processing.ImageChunker.chunks_to_array(segmentations, chunk_locations, image_array.shape)

        return np.squeeze(combined_segmentations)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if nnunet_log_filename is not None and nnunet_log_file is not None:
            nnunet_log_file.close()


def cropped_fov_prediction_pipeline(image, segmentation_array, workflow: models.ModelWorkflow, accelerator, nnunet_log_filename):
    """
    Process segmentation by resampling, limiting FOV, and predicting.

    Parameters:
        image (SimpleITK.Image): The input image.
        segmentation_array (np.array): The segmentation array to be processed.
        workflow (models.ModelWorkflow): List of routines where the second element contains model info.
        accelerator (any): The accelerator used for prediction.
        nnunet_log_filename (str): Path to the nnunet log file.

    Returns:
        model (str): The model name used in the process.
        segmentation_array (np.array): The final processed segmentation array.
    """
    # Get the second model from the routine
    model_to_crop_from = workflow[0]
    target_model = workflow[1]
    target_model_fov_information = target_model.limit_fov

    # Convert the segmentation array to SimpleITK image and set properties
    to_crop_segmentation = SimpleITK.GetImageFromArray(segmentation_array)
    to_crop_segmentation.SetOrigin(image.GetOrigin())
    to_crop_segmentation.SetSpacing(model_to_crop_from.voxel_spacing)
    to_crop_segmentation.SetDirection(image.GetDirection())

    # Resample the image using the desired spacing
    desired_spacing = target_model.voxel_spacing
    to_crop_image_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)
    to_crop_image = SimpleITK.GetImageFromArray(to_crop_image_array)
    to_crop_image.SetOrigin(image.GetOrigin())
    to_crop_image.SetSpacing(desired_spacing)
    to_crop_image.SetDirection(image.GetDirection())

    # Resample the segmentation
    resampled_to_crop_segmentation = image_processing.ImageResampler.resample_segmentation(to_crop_image, to_crop_segmentation)
    del to_crop_segmentation
    resampled_to_crop_segmentation_array = SimpleITK.GetArrayFromImage(resampled_to_crop_segmentation)

    # Limit FOV based on model information
    limited_fov_image_array, original_fov_info = image_processing.limit_fov(to_crop_image_array, resampled_to_crop_segmentation_array,
                                                                            target_model_fov_information["inference_fov_intensities"])

    to_write_image = SimpleITK.GetImageFromArray(limited_fov_image_array)
    to_write_image.SetOrigin(image.GetOrigin())
    to_write_image.SetSpacing(desired_spacing)
    to_write_image.SetDirection(image.GetDirection())

    # Predict the limited FOV segmentation
    limited_fov_segmentation_array = predict_from_array_by_iterator(limited_fov_image_array, target_model,
                                                                            accelerator, nnunet_log_filename)

    # Expand the segmentation to the original FOV
    expanded_segmentation_array = image_processing.expand_segmentation_fov(limited_fov_segmentation_array, original_fov_info)

    # Limit the FOV again based on label intensities and largest component condition
    limited_fov_segmentation_array, original_fov_info = image_processing.limit_fov(expanded_segmentation_array, resampled_to_crop_segmentation_array,
                                                                                   target_model_fov_information["label_intensity_to_crop_from"],
                                                                                   target_model_fov_information["largest_component_only"])

    # Expand the segmentation array to the original FOV
    segmentation_array = image_processing.expand_segmentation_fov(limited_fov_segmentation_array, original_fov_info)

    # Return the segmentation array and spacing
    return segmentation_array, desired_spacing