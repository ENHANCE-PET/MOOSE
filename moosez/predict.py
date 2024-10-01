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
from moosez import models
from moosez import image_processing
from moosez.resources import check_device
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


def predict_from_array_by_iterator(image_array: np.ndarray, model: models.Model, accelerator: str = None, nnunet_log_filename: str = None):
    image_array = image_array[None, ...]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if nnunet_log_filename is not None:
        nnunet_log_file = open(nnunet_log_filename, "a")
        sys.stdout = nnunet_log_file
        sys.stderr = nnunet_log_file

    try:
        if accelerator is None:
            accelerator = check_device()

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
        if nnunet_log_filename is not None:
            nnunet_log_file.close()
