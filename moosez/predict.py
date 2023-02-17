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
import subprocess
from halo import Halo

from moosez import constants


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
        return 208
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


def predict(model_name: str, input_dir: str, output_dir: str):
    """
    Runs the prediction using nnunet_predict.
    :param model_name: The name of the model.
    :param input_dir: The input directory.
    :param output_dir: The output directory.
    :return: None
    """
    task_number = map_model_name_to_task_number(model_name)
    # set the environment variables
    os.environ["RESULTS_FOLDER"] = constants.NNUNET_RESULTS_FOLDER
    subprocess.run(f'nnUNet_predict -i {input_dir} -o {output_dir} -t {task_number} -m 3d_fullres -f all',
                   shell=True, env=os.environ, capture_output=True)


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


def monitor_output_directory(input_dir, output_dir, total_files, spinner):
    """
    Continuously monitors the specified output directory for new files and updates the progress bar accordingly.
    Parameters:
        input_dir (str): The path to the input directory.
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
            spinner.text = f'From {os.path.basename(input_dir)} MOOSE processed {files_processed} of {total_files} ' \
                           f'files...'
            spinner.spinner = 'dots'
        files_processed = new_files_processed
