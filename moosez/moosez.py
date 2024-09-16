#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# The main module of the mooseZ. It contains the main function that is executed when the mooseZ is run.
#
# Usage:
# The main function in this module is executed when the mooseZ is run.
#
# ----------------------------------------------------------------------------------------------------------------------
import os

os.environ["nnUNet_raw"] = ""
os.environ["nnUNet_preprocessed"] = ""
os.environ["nnUNet_results"] = ""

import argparse
import logging
import time
from datetime import datetime
import SimpleITK
import colorama
import emoji
import numpy
from halo import Halo
from moosez import constants
from moosez import display
from moosez import file_utilities
from moosez import image_conversion
from moosez import image_processing
from moosez import input_validation
from moosez import predict
from moosez import resources
from moosez import models
from moosez.image_processing import ImageResampler
from moosez.nnUNet_custom_trainer.utility import add_custom_trainers_to_local_nnunetv2
from moosez.resources import AVAILABLE_MODELS

import threading
from moosez.benchmarking.utilities import PerformanceObserver


def main():
    colorama.init()

    # Argument parser
    parser = argparse.ArgumentParser(
        description=display.get_usage_message(),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    # Main directory containing subject folders
    parser.add_argument(
        "-d", "--main_directory",
        type=str,
        required=True,
        metavar="<MAIN_DIRECTORY>",
        help="Specify the main directory containing subject folders."
    )

    # Name of the models to use for segmentation
    parser.add_argument(
        "-m", "--model_names",
        nargs='+',
        type=str,
        choices=AVAILABLE_MODELS,
        required=True,
        metavar="<MODEL_NAMES>",
        help="Choose the models for segmentation from the following:\n" + "\n".join(AVAILABLE_MODELS)
    )

    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        default=False,
        help="Activate benchmarking."
    )

    # Custom help option
    parser.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )

    args = parser.parse_args()
    parent_folder = os.path.abspath(args.main_directory)
    model_names = args.model_names
    benchmark = args.benchmark

    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                        filename=os.path.join(parent_folder, datetime.now().strftime('moosez-v.3.0.0_%H-%M-%d-%m-%Y.log')), filemode='w')
    nnunet_log_filename = os.path.join(parent_folder, datetime.now().strftime('nnunet_%H-%M-%d-%m-%Y.log'))

    display.logo()
    display.citation()

    logging.info('----------------------------------------------------------------------------------------------------')
    logging.info('                                     STARTING MOOSE-Z V.3.0.0                                       ')
    logging.info('----------------------------------------------------------------------------------------------------')

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    logging.info(' ')
    logging.info('- Main directory: ' + parent_folder)
    logging.info(' ')
    print(' ')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    print(' ')
    modalities = display.expectations(model_names)
    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    logging.info('- Custom trainer: ' + custom_trainer_status)
    accelerator = resources.check_device()

    # ----------------------------------
    # DOWNLOADING THE MODEL
    # ----------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    print('')
    model_path = resources.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(model_names)

    # ----------------------------------
    # INPUT STANDARDIZATION
    # ----------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} STANDARDIZING INPUT DATA TO NIFTI:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' STANDARDIZING INPUT DATA TO NIFTI:')
    logging.info(' ')
    image_conversion.standardize_to_nifti(parent_folder)
    print(f"{constants.ANSI_GREEN} Standardization complete.{constants.ANSI_RESET}")
    logging.info(" Standardization complete.")

    # --------------------------------------
    # CHECKING FOR MOOSE COMPLIANT SUBJECTS
    # --------------------------------------

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    moose_compliant_subjects = input_validation.select_moose_compliant_subjects(subjects, modalities)

    num_subjects = len(moose_compliant_subjects)

    if num_subjects < 1:
        print(f'{constants.ANSI_RED} {emoji.emojize(":cross_mark:")} No moose compliant subject found to continue!{constants.ANSI_RESET} {emoji.emojize(":light_bulb:")} See: https://github.com/ENHANCE-PET/MOOSE#directory-structure-and-naming-conventions-for-moose-%EF%B8%8F')
        return

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR MOOSE COMPLIANT SUBJECTS
    # -------------------------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":crystal_ball:")} PREDICT:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' PERFORMING PREDICTION:')

    spinner = Halo(text=' Initiating', spinner='dots')
    spinner.start()
    start_total_time = time.time()

    for i, subject in enumerate(moose_compliant_subjects):
        # SETTING UP DIRECTORY STRUCTURE
        subject_name = os.path.basename(subject)
        logging.info(' ')
        logging.info(f' SUBJECT: {subject_name}')

        if benchmark:
            performance_observer = PerformanceObserver(subject_name, ', '.join(model_names))
            monitoring_thread = threading.Thread(target=performance_observer.monitor_memory_usage, args=(0.1, ))
            monitoring_thread.start()

        spinner.text = f'[{i + 1}/{num_subjects}] Setting up directory structure for {subject_name}...'
        logging.info(' ')
        logging.info(f' SETTING UP MOOSE-Z DIRECTORY:')
        logging.info(' ')
        moose_dir, segmentations_dir, stats_dir = file_utilities.moose_folder_structure(subject)
        logging.info(f" MOOSE directory for subject {subject_name} at: {moose_dir}")

        # RUN PREDICTION
        start_time = time.time()
        logging.info(' ')
        logging.info(' RUNNING PREDICTION:')
        logging.info(' ')

        performance_observer.record_phase("Loading Image")
        file_path = file_utilities.get_files(subject, '.nii.gz')[0]
        image = SimpleITK.ReadImage(file_path)
        file_name = file_utilities.get_nifti_file_stem(file_path)
        pet_file = file_utilities.find_pet_file(subject)
        performance_observer.time_phase()

        for desired_spacing, model_workflows in model_routine.items():
            performance_observer.record_phase(f"Resampling Image: {'x'.join(map(str,desired_spacing))}")
            resampling_time_start = time.time()
            resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)
            logging.info(f' - Resampling at {"x".join(map(str,desired_spacing))} took: {round((time.time() - resampling_time_start), 2)}s')
            performance_observer.time_phase()

            for model_workflow in model_workflows:
                performance_observer.record_phase(f"Predicting: {model_workflow.target_model}")
                # ----------------------------------
                # RUN MODEL WORKFLOW
                # ----------------------------------
                model_time_start = time.time()
                spinner.text = f'[{i + 1}/{num_subjects}] Running prediction for {subject_name} using {model_workflow[0]}...'
                logging.info(f'   - Model {model_workflow.target_model}')
                segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model_workflow[0], accelerator, nnunet_log_filename)

                if len(model_workflow) == 2:
                    inference_fov_intensities = model_workflow[1].limit_fov["inference_fov_intensities"]
                    if isinstance(inference_fov_intensities, int):
                        inference_fov_intensities = [inference_fov_intensities]

                    existing_intensities = numpy.unique(segmentation_array)
                    if not all([intensity in existing_intensities for intensity in inference_fov_intensities]):
                        spinner.text = f'[{i + 1}/{num_subjects}] Organ to crop from not in initial FOV...'
                        logging.info("     - Organ to crop from not in initial FOV.")
                        continue

                    segmentation_array, desired_spacing = image_processing.cropped_fov_prediction_pipeline(image, segmentation_array, model_workflow, accelerator, nnunet_log_filename)

                segmentation = SimpleITK.GetImageFromArray(segmentation_array)
                segmentation.SetSpacing(desired_spacing)
                segmentation.SetOrigin(image.GetOrigin())
                segmentation.SetDirection(image.GetDirection())
                resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

                segmentation_image_path = os.path.join(segmentations_dir, f"{model_workflow.target_model.multilabel_prefix}segmentation_{file_name}.nii.gz")
                logging.info(f'     - Writing segmentation for {model_workflow.target_model}')
                SimpleITK.WriteImage(resampled_segmentation, segmentation_image_path)
                logging.info(f"     - Prediction complete for {model_workflow.target_model} within {round((time.time() - model_time_start)/ 60, 1)} min.")

                # ----------------------------------
                # EXTRACT VOLUME STATISTICS
                # ----------------------------------
                spinner.text = f'[{i + 1}/{num_subjects}] Extracting CT volume statistics for {subject_name} ({model_workflow.target_model})...'
                logging.info(f'     - Extracting volume statistics for {model_workflow.target_model}')
                out_csv = os.path.join(stats_dir, model_workflow.target_model.multilabel_prefix + subject_name + '_ct_volume.csv')
                image_processing.get_shape_statistics(resampled_segmentation, model_workflow.target_model, out_csv)
                spinner.text = f'{constants.ANSI_GREEN} [{i + 1}/{num_subjects}] CT volume extracted for {subject_name}! ' \
                               f'{constants.ANSI_RESET}'
                time.sleep(1)

                # ----------------------------------
                # EXTRACT PET ACTIVITY
                # ----------------------------------
                if pet_file is not None:
                    pet_image = SimpleITK.ReadImage(pet_file)
                    spinner.text = f'[{i + 1}/{num_subjects}] Extracting PET activity for {subject_name} ({model_workflow.target_model})...'
                    logging.info(f'     - Extracting PET statistics for {model_workflow.target_model}')
                    resampled_multilabel_image = ImageResampler.reslice_identity(reference_image=pet_image,
                                                                                 moving_image=resampled_segmentation,
                                                                                 is_label_image=True)
                    out_csv = os.path.join(stats_dir, model_workflow.target_model.multilabel_prefix + subject_name + '_pet_activity.csv')
                    image_processing.get_intensity_statistics(pet_image, resampled_multilabel_image, model_workflow.target_model, out_csv)
                    spinner.text = f'{constants.ANSI_GREEN} [{i + 1}/{num_subjects}] PET activity extracted for {subject_name}! ' \
                                   f'{constants.ANSI_RESET}'
                    time.sleep(3)

                performance_observer.time_phase()

        end_time = time.time()
        elapsed_time = end_time - start_time
        spinner.text = f' {constants.ANSI_GREEN}[{i + 1}/{num_subjects}] Prediction done for {subject_name} using {len(model_names)} models: ' \
                       f' | Elapsed time: {round(elapsed_time / 60, 1)} min{constants.ANSI_RESET}'
        time.sleep(3)
        logging.info(f' Prediction done for {subject_name} using {len(model_names)} models: {", ".join(model_names)}!' 
                     f' | Elapsed time: {round(elapsed_time / 60, 1)} min')

        performance_observer.record_phase("Total Processing Done")
        if benchmark:
            performance_observer.off()
            monitoring_thread.join()
            output_path = os.path.join(stats_dir, '')
            performance_observer.plot_performance(output_path)

    end_total_time = time.time()
    total_elapsed_time = (end_total_time - start_total_time) / 60
    time_per_dataset = total_elapsed_time / len(moose_compliant_subjects)

    spinner.succeed(f'{constants.ANSI_GREEN} All predictions done! | Total elapsed time for '
                    f'{len(moose_compliant_subjects)} datasets, {len(model_names)} models: {round(total_elapsed_time, 1)} min'
                    f' | Time per dataset: {round(time_per_dataset, 2)} min {constants.ANSI_RESET}')
    logging.info(f' ')
    logging.info(f' ALL SUBJECTS PROCESSED')
    logging.info(f'  - Number of Subjects: {len(moose_compliant_subjects)}')
    logging.info(f'  - Number of Models:   {len(model_names)}')
    logging.info(f'  - Time (total):       {round(total_elapsed_time, 1)}min')
    logging.info(f'  - Time (per subject): {round(time_per_dataset, 2)}min')
    logging.info('----------------------------------------------------------------------------------------------------')
    logging.info('                                     FINISHED MOOSE-Z V.3.0.0                                       ')
    logging.info('----------------------------------------------------------------------------------------------------')


def moose(input_data: str | tuple[numpy.ndarray, tuple[float, float, float]] | SimpleITK.Image, model_names: str | list[str], output_dir: str = None, accelerator: str = None) -> None:
    """
    Execute the MOOSE 3.0 image segmentation process.

    This function carries out the following steps:
    1. Sets the path for model results.
    2. Creates the required directory for the model.
    3. Downloads the model based on the provided `model_name`.
    4. Validates and prepares the input directory to be compatible with nnUNet.
    5. Executes the prediction process.

    :param input_data: This can be either:
                       1. A file path to the NIfTI file (as a string),
                       2. A tuple containing a numpy array and the corresponding spacing (as (array, spacing)),
                       3. A SimpleITK image.

    :param model_names: The name of the model to be used for predictions. This model will be downloaded and used
                       for the image segmentation process.
    :type model_names: str

    :param output_dir: Path to the directory where the segmented output will be saved.
    :type output_dir: str

    :param accelerator: Specifies the type of accelerator to be used. Common values include "cpu" and "cuda" for
                        GPU acceleration.
    :type accelerator: str

    :return: None
    :rtype: None

    :Example:

    >>> moose('/path/to/input/file', '[list, of, models]', '/path/to/save/output', 'cuda')
    >>> moose((numpy_array, (1.5, 1.5, 1.5)), 'model_name', '/path/to/save/output', 'cuda')
    >>> moose(simple_itk_image, 'model_name', '/path/to/save/output', 'cuda')

    """

    if isinstance(input_data, str):
        image = SimpleITK.ReadImage(input_data)
        file_name = file_utilities.get_nifti_file_stem(input_data)
    elif isinstance(input_data, SimpleITK.Image):
        image = input_data
        file_name = 'image_from_simpleitk'  # Default name or generate from other means if needed
    elif isinstance(input_data, tuple) and isinstance(input_data[0], numpy.ndarray) and isinstance(input_data[1],
                                                                                                   tuple):
        numpy_array, spacing = input_data
        image = SimpleITK.GetImageFromArray(numpy_array)
        image.SetSpacing(spacing)
        file_name = 'image_from_array'  # Default name or generate from other means if needed
    else:
        raise ValueError(
            "Input data must be either a file path (str), a SimpleITK.Image, or a tuple (numpy array, spacing).")

    if isinstance(model_names, str):
        model_names = [model_names]

    model_path = resources.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(model_names)

    for desired_spacing, model_workflows in model_routine.items():
        resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline',
                                                                                              desired_spacing)

        for model_workflow in model_workflows:
            segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model_workflow[0], accelerator,
                                                                        os.devnull)

            if len(model_workflow) == 2:
                inference_fov_intensities = model_workflow[1].limit_fov["inference_fov_intensities"]
                if isinstance(inference_fov_intensities, int):
                    inference_fov_intensities = [inference_fov_intensities]

                existing_intensities = numpy.unique(segmentation_array)
                if not all([intensity in existing_intensities for intensity in inference_fov_intensities]):
                    continue

                segmentation_array, desired_spacing = image_processing.cropped_fov_prediction_pipeline(image,
                                                                                                       segmentation_array,
                                                                                                       model_workflow,
                                                                                                       accelerator,
                                                                                                       os.devnull)

            segmentation = SimpleITK.GetImageFromArray(segmentation_array)
            segmentation.SetSpacing(desired_spacing)
            segmentation.SetOrigin(image.GetOrigin())
            segmentation.SetDirection(image.GetDirection())
            resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

            if output_dir is None:
                output_dir = os.path.dirname(input_data) if isinstance(input_data, str) else '.'
            segmentation_image_path = os.path.join(output_dir, f"{model_workflow.target_model.multilabel_prefix}segmentation_{file_name}.nii.gz")
            SimpleITK.WriteImage(resampled_segmentation, segmentation_image_path)


if __name__ == '__main__':
    main()
