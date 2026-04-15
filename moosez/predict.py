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
import torch
import numpy as np
from typing import Tuple, List, Dict, Iterator
from moosez import models
from moosez import image_processing
from moosez import system
from moosez import constants
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
    predictor.initialize_from_trained_model_folder(model.configuration_directory, use_folds=model.folds)
    return predictor


@dask.delayed
def process_case(preprocessor, chunk: np.ndarray, chunk_properties: Dict, predictor: nnUNetPredictor, location: Tuple) -> Dict:
    data, seg, prop = preprocessor.run_case_npy(chunk,
                                                None,
                                                chunk_properties,
                                                predictor.plans_manager,
                                                predictor.configuration_manager,
                                                predictor.dataset_json)

    data_tensor = torch.from_numpy(data).contiguous()
    if predictor.device.type == "cuda":
        data_tensor = data_tensor.pin_memory()

    return {'data': data_tensor, 'data_properties': prop, 'ofile': None, 'location': location}


def preprocessing_iterator_from_array(image_array: np.ndarray, image_properties: Dict, predictor: nnUNetPredictor, output_manager: system.OutputManager) -> Tuple[Iterator, List[Dict]]:
    splits = image_processing.ImageChunker.determine_splits(image_array)
    chunks, locations = image_processing.ImageChunker.array_to_chunks(image_array, splits, constants.OVERLAP_PER_AXIS)
    chunk_properties = [image_properties.copy() for _ in chunks]

    output_manager.log_update(f"     - Image split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        output_manager.log_update(f"       - {i + 1}: {'x'.join(map(str, chunk.shape))}")

    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=True)

    delayed_tasks = []
    for image_chunk, chunk_property, location in zip(chunks, chunk_properties, locations):
        delayed_task = dask.delayed(process_case)(preprocessor, image_chunk, chunk_property, predictor, location)
        delayed_tasks.append(delayed_task)

    results = dask.compute(*delayed_tasks)
    iterator = iter(results)

    return iterator, locations


def predict_from_array_by_iterator(image_array: np.ndarray, model: models.Model, accelerator: str, output_manager: system.OutputManager) -> np.ndarray:
    image_array = image_array[None, ...]

    with output_manager.manage_nnUNet_output():
        predictor = initialize_predictor(model, accelerator)
        image_properties = {'spacing': model.voxel_spacing}

        iterator, chunk_locations = preprocessing_iterator_from_array(image_array, image_properties, predictor, output_manager)
        segmentations = predictor.predict_from_data_iterator(iterator)
        output_manager.log_update(f"     - Retrieved {len(segmentations)} chunks")
        segmentations = [segmentation[None, ...] for segmentation in segmentations]
        combined_segmentations = image_processing.ImageChunker.chunks_to_array(segmentations, chunk_locations, image_array.shape)
        output_manager.log_update(f"     - Combined them to an {'x'.join(map(str, combined_segmentations.shape))} array")

    return np.squeeze(combined_segmentations)
