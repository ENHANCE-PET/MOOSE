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
import time
import SimpleITK
import colorama
import emoji
import numpy
import pandas as pd
import multiprocessing as mp
import concurrent.futures
from moosez import constants
from moosez import download
from moosez import file_utilities
from moosez import image_conversion
from moosez import image_processing
from moosez import input_validation
from moosez import predict
from moosez import system
from moosez import models
from moosez.nnUNet_custom_trainer.utility import add_custom_trainers_to_local_nnunetv2
from moosez.benchmarking.benchmark import PerformanceObserver


def main():
    colorama.init()

    # Argument parser
    parser = argparse.ArgumentParser(
        description=constants.USAGE_MESSAGE,
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    # Main directory containing subject folders
    parser.add_argument(
        "-d", "--main_directory",
        type=str,
        metavar="<MAIN_DIRECTORY>",
        help="Specify the main directory containing subject folders."
    )

    # Name of the models to use for segmentation
    parser.add_argument(
        "-m", "--model_names",
        nargs='+',
        type=str,
        choices=models.AVAILABLE_MODELS,
        metavar="<MODEL_NAMES>",
        help="Choose the models for segmentation from the following:\n" + "\n".join(models.AVAILABLE_MODELS)
    )

    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        default=False,
        help="Activate benchmarking."
    )

    parser.add_argument(
        "-v-off", "--verbose_off",
        action="store_false",
        help="Deactivate verbose console."
    )

    parser.add_argument(
        "-log-off", "--logging_off",
        action="store_false",
        help="Deactivate logging."
    )

    parser.add_argument(
        '-herd', '--moose_herd',
        nargs='?',
        const=2,
        type=int,
        help='Specify the concurrent jobs (default: 2)'
    )

    parser.add_argument(
        '-dtd', '--download_training_data',
        action="store_true",
        default=False,
        help='Download the enhance 1.6k ENHANCE dataset'
    )

    parser.add_argument(
        '-dd', '--download_directory',
        default=None,
        type=str,
        help='Path to save the enhance 1.6k ENHANCE dataset'
    )

    # Custom help option
    parser.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )

    args = parser.parse_args()

    # ----------------------------------
    # OUTPUT SETTINGS
    # ----------------------------------

    verbose_console = args.verbose_off
    verbose_log = args.logging_off

    if args.download_training_data:
        verbose_console = True
        verbose_log = False

    output_manager = system.OutputManager(verbose_console, verbose_log)
    output_manager.display_logo()
    output_manager.display_authors()
    output_manager.display_doi()

    # ----------------------------------
    # DOWNLOADING THE ENHANCE DATA
    # ----------------------------------

    if args.download_training_data:
        output_manager.console_update(f'')
        output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} ENHANCE 1.6k DATA DOWNLOAD:{constants.ANSI_RESET}')
        output_manager.console_update(f'')
        download.download_enhance_data(args.download_directory, output_manager)
        return

    # ----------------------------------
    # START MOOSE
    # ----------------------------------

    parent_folder = os.path.abspath(args.main_directory)
    model_names = args.model_names
    benchmark = args.benchmark
    moose_instances = args.moose_herd

    output_manager.configure_logging(parent_folder)
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update('                                     STARTING MOOSE-Z V.3.0.0                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')


    # ----------------------------------
    # DOWNLOADING THE MODEL
    # ----------------------------------

    output_manager.console_update(f'')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    output_manager.console_update(f'')
    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(model_names, output_manager)

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    output_manager.log_update(f' ')
    output_manager.log_update(f'- Main directory: {parent_folder}')
    output_manager.log_update(f' ')
    output_manager.console_update(f' ')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    output_manager.console_update(f' ')

    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    modalities = input_validation.determine_model_expectations(model_routine, output_manager)
    output_manager.log_update(f'- Custom trainer: {custom_trainer_status}')
    accelerator, device_count = system.check_device(output_manager)
    if moose_instances is not None:
        output_manager.console_update(f" Number of moose instances run in parallel: {moose_instances}")

    # ----------------------------------
    # INPUT STANDARDIZATION
    # ----------------------------------

    output_manager.console_update(f'')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} STANDARDIZING INPUT DATA TO NIFTI:{constants.ANSI_RESET}')
    output_manager.console_update(f'')
    output_manager.log_update(f' ')
    output_manager.log_update(f' STANDARDIZING INPUT DATA TO NIFTI:')
    output_manager.log_update(f' ')
    image_conversion.standardize_to_nifti(parent_folder, output_manager)
    output_manager.console_update(f"{constants.ANSI_GREEN} Standardization complete.{constants.ANSI_RESET}")
    output_manager.log_update(f" Standardization complete.")

    # --------------------------------------
    # CHECKING FOR MOOSE COMPLIANT SUBJECTS
    # --------------------------------------

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    moose_compliant_subjects = input_validation.select_moose_compliant_subjects(subjects, modalities, output_manager)

    num_subjects = len(moose_compliant_subjects)

    if num_subjects < 1:
        output_manager.console_update(f'{constants.ANSI_RED} {emoji.emojize(":cross_mark:")} No moose compliant subject found to continue!{constants.ANSI_RESET} {emoji.emojize(":light_bulb:")} See: https://github.com/ENHANCE-PET/MOOSE#directory-structure-and-naming-conventions-for-moose-%EF%B8%8F')
        return

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR MOOSE COMPLIANT SUBJECTS
    # -------------------------------------------------

    output_manager.console_update(f'')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":crystal_ball:")} PREDICT:{constants.ANSI_RESET}')
    output_manager.console_update(f'')
    output_manager.log_update(' ')
    output_manager.log_update(' PERFORMING PREDICTION:')

    output_manager.spinner_start(' Initiating')
    start_total_time = time.time()

    subject_performance_parameters = []

    if moose_instances is not None:
        output_manager.log_update(f"- Branching out with {moose_instances} concurrent jobs.")

        performance_observer = PerformanceObserver(f'All {num_subjects} subjects | {moose_instances} jobs', ', '.join(model_names))
        if benchmark:
            performance_observer.on()

        mp_context = mp.get_context('spawn')
        processed_subjects = 0
        output_manager.spinner_update(f'[{processed_subjects}/{num_subjects}] subjects processed.')

        if device_count is not None and device_count > 1:
            accelerator_assignments = [f"{accelerator}:{i % device_count}" for i in range(len(subjects))]
        else:
            accelerator_assignments = [accelerator] * len(subjects)

        with concurrent.futures.ProcessPoolExecutor(max_workers=moose_instances, mp_context=mp_context) as executor:
            futures = []
            for i, (subject, accelerator) in enumerate(zip(moose_compliant_subjects, accelerator_assignments)):
                futures.append(executor.submit(moose_subject, subject, i, num_subjects,
                                               model_routine, accelerator,
                                               None, benchmark))

            for future in concurrent.futures.as_completed(futures):
                if benchmark:
                    subject_performance_parameters.append(future.result())
                processed_subjects += 1
                output_manager.spinner_update(f'[{processed_subjects}/{num_subjects}] subjects processed.')

        performance_observer.record_phase("Total Processing Done")
        if benchmark:
            performance_observer.off()
            subject_performance_parameters.append(performance_observer.get_peak_resources())

    else:
        for i, subject in enumerate(moose_compliant_subjects):
            subject_performance = moose_subject(subject, i, num_subjects,
                                                model_routine, accelerator,
                                                output_manager, benchmark)
            if benchmark:
                subject_performance_parameters.append(subject_performance)

    end_total_time = time.time()
    total_elapsed_time = (end_total_time - start_total_time) / 60
    time_per_dataset = total_elapsed_time / len(moose_compliant_subjects)
    time_per_model = time_per_dataset / len(model_names)

    output_manager.spinner_succeed(f'{constants.ANSI_GREEN} All predictions done! | Total elapsed time for '
                                   f'{len(moose_compliant_subjects)} datasets, {len(model_names)} models: {round(total_elapsed_time, 1)} min'
                                   f' | Time per dataset: {round(time_per_dataset, 2)} min'
                                   f' | Time per model: {round(time_per_model, 2)} min {constants.ANSI_RESET}')
    output_manager.log_update(f' ')
    output_manager.log_update(f' ALL SUBJECTS PROCESSED')
    output_manager.log_update(f'  - Number of Subjects: {len(moose_compliant_subjects)}')
    output_manager.log_update(f'  - Number of Models:   {len(model_names)}')
    output_manager.log_update(f'  - Time (total):       {round(total_elapsed_time, 1)}min')
    output_manager.log_update(f'  - Time (per subject): {round(time_per_dataset, 2)}min')
    output_manager.log_update(f'  - Time (per model):   {round(time_per_model, 2)}min')
    if benchmark:
        df = pd.DataFrame(subject_performance_parameters, columns=['Image', 'Model', 'Image Size', 'Runtime [s]', 'Peak Memory [GB]'])
        csv_file_path = os.path.join(parent_folder, 'moosez-v3.0.0_peak_performance_parameters.csv')
        df.to_csv(csv_file_path, index=False)
        output_manager.log_update(f'  - Resource utilization written to {csv_file_path}')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update('                                     FINISHED MOOSE-Z V.3.0.0                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')


def moose(input_data: str | tuple[numpy.ndarray, tuple[float, float, float]] | SimpleITK.Image, model_names: str | list[str], output_dir: str = None, accelerator: str = None) -> None:
    """
    Execute the MOOSE 3.0 image segmentation process.

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
        file_name = 'image_from_simpleitk'
    elif isinstance(input_data, tuple) and isinstance(input_data[0], numpy.ndarray) and isinstance(input_data[1],
                                                                                                   tuple):
        numpy_array, spacing = input_data
        image = SimpleITK.GetImageFromArray(numpy_array)
        image.SetSpacing(spacing)
        file_name = 'image_from_array'
    else:
        raise ValueError(
            "Input data must be either a file path (str), a SimpleITK.Image, or a tuple (numpy array, spacing).")

    if isinstance(model_names, str):
        model_names = [model_names]

    output_manager = system.OutputManager(False, False)

    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(model_names, output_manager)

    for desired_spacing, model_workflows in model_routine.items():
        resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)

        for model_workflow in model_workflows:
            segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model_workflow[0], accelerator, os.devnull)

            if len(model_workflow) == 2:
                inference_fov_intensities = model_workflow[1].limit_fov["inference_fov_intensities"]
                if isinstance(inference_fov_intensities, int):
                    inference_fov_intensities = [inference_fov_intensities]

                existing_intensities = numpy.unique(segmentation_array)
                if not all([intensity in existing_intensities for intensity in inference_fov_intensities]):
                    continue

                segmentation_array, desired_spacing = predict.cropped_fov_prediction_pipeline(image, segmentation_array,
                                                                                              model_workflow,
                                                                                              accelerator, os.devnull)

            segmentation = SimpleITK.GetImageFromArray(segmentation_array)
            segmentation.SetSpacing(desired_spacing)
            segmentation.SetOrigin(image.GetOrigin())
            segmentation.SetDirection(image.GetDirection())
            resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

            if output_dir is None:
                output_dir = os.path.dirname(input_data) if isinstance(input_data, str) else '.'
            segmentation_image_path = os.path.join(output_dir, f"{model_workflow.target_model.multilabel_prefix}segmentation_{file_name}.nii.gz")
            SimpleITK.WriteImage(resampled_segmentation, segmentation_image_path)


def moose_subject(subject: str, subject_index: int, number_of_subjects: int, model_routine: dict, accelerator: str,
                  output_manager: system.OutputManager | None, benchmark: bool = False):
    # SETTING UP DIRECTORY STRUCTURE
    subject_name = os.path.basename(subject)

    if output_manager is None:
        output_manager = system.OutputManager(False, False)

    output_manager.log_update(' ')
    output_manager.log_update(f' SUBJECT: {subject_name}')

    model_names = []
    for workflows in model_routine.values():
        for workflow in workflows:
            model_names.append(workflow.target_model.model_identifier)

    performance_observer = PerformanceObserver(subject_name, ', '.join(model_names))
    subject_peak_performance = None
    if benchmark:
        performance_observer.on()

    output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Setting up directory structure for {subject_name}...')
    output_manager.log_update(' ')
    output_manager.log_update(f' SETTING UP MOOSE-Z DIRECTORY:')
    output_manager.log_update(' ')
    moose_dir, segmentations_dir, stats_dir = file_utilities.moose_folder_structure(subject)
    output_manager.log_update(f" MOOSE directory for subject {subject_name} at: {moose_dir}")

    # RUN PREDICTION
    start_time = time.time()
    output_manager.log_update(' ')
    output_manager.log_update(' RUNNING PREDICTION:')
    output_manager.log_update(' ')

    performance_observer.record_phase("Loading Image")
    file_path = file_utilities.get_files(subject, 'CT_', ('.nii', '.nii.gz'))[0]
    image = image_processing.standardize_image(file_path, output_manager, moose_dir)
    file_name = file_utilities.get_nifti_file_stem(file_path)
    pet_file = file_utilities.find_pet_file(subject)
    performance_observer.metadata_image_size = image.GetSize()
    performance_observer.time_phase()

    for desired_spacing, model_workflows in model_routine.items():
        performance_observer.record_phase(f"Resampling Image: {'x'.join(map(str,desired_spacing))}")
        resampling_time_start = time.time()
        resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)
        output_manager.log_update(f' - Resampling at {"x".join(map(str,desired_spacing))} took: {round((time.time() - resampling_time_start), 2)}s')
        performance_observer.time_phase()

        for model_workflow in model_workflows:
            performance_observer.record_phase(f"Predicting: {model_workflow.target_model}")
            # ----------------------------------
            # RUN MODEL WORKFLOW
            # ----------------------------------
            model_time_start = time.time()
            output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Running prediction for {subject_name} using {model_workflow[0]}...')
            output_manager.log_update(f'   - Model {model_workflow.target_model}')
            segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model_workflow[0], accelerator, output_manager.nnunet_log_filename)

            if len(model_workflow) == 2:
                inference_fov_intensities = model_workflow[1].limit_fov["inference_fov_intensities"]
                if isinstance(inference_fov_intensities, int):
                    inference_fov_intensities = [inference_fov_intensities]

                existing_intensities = numpy.unique(segmentation_array)
                if not all([intensity in existing_intensities for intensity in inference_fov_intensities]):
                    output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Organ to crop from not in initial FOV...')
                    output_manager.log_update("     - Organ to crop from not in initial FOV.")
                    performance_observer.time_phase()
                    continue

                segmentation_array, desired_spacing = predict.cropped_fov_prediction_pipeline(image, segmentation_array, model_workflow, accelerator, output_manager.nnunet_log_filename)

            segmentation = SimpleITK.GetImageFromArray(segmentation_array)
            segmentation.SetSpacing(desired_spacing)
            segmentation.SetOrigin(image.GetOrigin())
            segmentation.SetDirection(image.GetDirection())
            resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

            segmentation_image_path = os.path.join(segmentations_dir, f"{model_workflow.target_model.multilabel_prefix}segmentation_{file_name}.nii.gz")
            output_manager.log_update(f'     - Writing segmentation for {model_workflow.target_model}')
            SimpleITK.WriteImage(resampled_segmentation, segmentation_image_path)
            output_manager.log_update(f"     - Prediction complete for {model_workflow.target_model} within {round((time.time() - model_time_start)/ 60, 1)} min.")

            # ----------------------------------
            # EXTRACT VOLUME STATISTICS
            # ----------------------------------
            output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Extracting CT volume statistics for {subject_name} ({model_workflow.target_model})...')
            output_manager.log_update(f'     - Extracting volume statistics for {model_workflow.target_model}')
            out_csv = os.path.join(stats_dir, model_workflow.target_model.multilabel_prefix + subject_name + '_ct_volume.csv')
            image_processing.get_shape_statistics(resampled_segmentation, model_workflow.target_model, out_csv)
            output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] CT volume extracted for {subject_name}! {constants.ANSI_RESET}')
            time.sleep(1)

            # ----------------------------------
            # EXTRACT PET ACTIVITY
            # ----------------------------------
            if pet_file is not None:
                pet_image = SimpleITK.ReadImage(pet_file)
                output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Extracting PET activity for {subject_name} ({model_workflow.target_model})...')
                output_manager.log_update(f'     - Extracting PET statistics for {model_workflow.target_model}')
                resampled_multilabel_image = image_processing.ImageResampler.reslice_identity(pet_image, resampled_segmentation, is_label_image=True)
                out_csv = os.path.join(stats_dir, model_workflow.target_model.multilabel_prefix + subject_name + '_pet_activity.csv')
                image_processing.get_intensity_statistics(pet_image, resampled_multilabel_image, model_workflow.target_model, out_csv)
                output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] PET activity extracted for {subject_name}! {constants.ANSI_RESET}')
                time.sleep(1)

            performance_observer.time_phase()

    end_time = time.time()
    elapsed_time = end_time - start_time
    output_manager.spinner_update(f' {constants.ANSI_GREEN}[{subject_index + 1}/{number_of_subjects}] Prediction done for {subject_name} using {len(model_names)} models: '
                                  f' | Elapsed time: {round(elapsed_time / 60, 1)} min{constants.ANSI_RESET}')
    time.sleep(1)
    output_manager.log_update(f' Prediction done for {subject_name} using {len(model_names)} models: {", ".join(model_names)}!' 
                              f' | Elapsed time: {round(elapsed_time / 60, 1)} min')

    performance_observer.record_phase("Total Processing Done")
    if benchmark:
        performance_observer.off()
        performance_observer.plot_performance(stats_dir)
        subject_peak_performance = performance_observer.get_peak_resources()

    return subject_peak_performance


if __name__ == '__main__':
    main()
