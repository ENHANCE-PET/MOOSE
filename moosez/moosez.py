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
import argparse
import time
import SimpleITK
import colorama
import emoji
import numpy
import pandas as pd
import multiprocessing as mp
import concurrent.futures
from typing import Union, Tuple, List, Iterator
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
        "-sl", "--single_labels",
        action="store_true",
        help="Writes all labels of each segmentation as single label image with the intensity 1 if activated."
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

    parser.add_argument(
        '-md', '--model_download',
        nargs='+',
        type=str,
        choices=models.AVAILABLE_MODELS,
        metavar="<MODEL_NAMES>",
        help='Download one or more models from the model zoo without running segmentation.'
    )

    parser.add_argument(
        '-md-out', '--model_download_directory',
        type=str,
        default=None,
        help='Specify a custom directory to dump downloaded models.'
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
    # DOWNLOADING THE MODEL
    # ----------------------------------

    if args.model_download:
        output_manager.console_update(f'\n{constants.ANSI_VIOLET} {emoji.emojize(":package:")} INITIATING MODEL(S) DOWNLOAD: {constants.ANSI_RESET}\n')
        output_manager.console_update("")

        # Check whether user provided a custom path
        using_default_path = args.model_download_directory is None
        custom_root = os.path.abspath(args.model_download_directory or system.MODELS_DIRECTORY_PATH)

        # Avoid double nesting if already models/nnunet_trained_models
        if os.path.basename(custom_root) == "nnunet_trained_models" and os.path.basename(os.path.dirname(custom_root)) == "models":
            model_output_path = custom_root
        else:
            model_output_path = os.path.join(custom_root, "models", "nnunet_trained_models")

        if using_default_path:
            output_manager.console_update(f' {emoji.emojize(":warning:")} No model output path specified. Using default: {constants.ANSI_ORANGE} {model_output_path}{constants.ANSI_RESET}')

        for model_name in args.model_download:
            if not models.Model.model_identifier_valid(model_name, output_manager):
                output_manager.console_update(f"{constants.ANSI_RED} ✖ Invalid model: {model_name}{constants.ANSI_RESET}")
                continue
            models.Model(model_name, output_manager, base_directory=model_output_path)

        # Docker bind hint
        docker_bind = os.path.abspath(os.path.join(model_output_path, ".."))  # parent of nnunet_trained_models
        example_model = args.model_download[0]
        output_manager.console_update("")
        output_manager.display_docker_usage(docker_bind, example_model)
        return

    # ----------------------------------
    # START MOOSE
    # ----------------------------------

    parent_folder = os.path.abspath(args.main_directory)
    model_names = args.model_names
    benchmark = args.benchmark
    moose_instances = args.moose_herd
    single_labels = args.single_labels

    output_manager.configure_logging(parent_folder)
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update(f'                                     STARTING MOOSE-Z v{system.MOOSE_VERSION}                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')

    # ----------------------------------
    # DOWNLOADING THE MODEL
    # ----------------------------------

    output_manager.console_update(f'')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    output_manager.console_update(f'')
    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_workflows = models.construct_model_workflows(model_names, output_manager)

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    output_manager.log_update(f' ')
    output_manager.log_update(f'- Main directory: {parent_folder}')
    output_manager.log_update(f' ')
    output_manager.console_update(f' ')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    output_manager.console_update(f' ')

    modalities = input_validation.determine_model_expectations(model_workflows, output_manager)
    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
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
                futures.append(executor.submit(moose_subject, subject, i, num_subjects, model_workflows,
                                               accelerator, None, benchmark, single_labels))

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
            subject_performance = moose_subject(subject, i, num_subjects, model_workflows,
                                                accelerator, output_manager, benchmark, single_labels)
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
        csv_file_path = os.path.join(parent_folder, f'moosez-v{system.MOOSE_VERSION}_peak_performance_parameters.csv')
        df.to_csv(csv_file_path, index=False)
        output_manager.log_update(f'  - Resource utilization written to {csv_file_path}')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update(f'                                     FINISHED MOOSE-Z v{system.MOOSE_VERSION}                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')


def moose(input_data: Union[str, Tuple[numpy.ndarray, Tuple[float, float, float]], SimpleITK.Image],
          model_names: Union[str, List[str]], output_dir: str = None, accelerator: str = None) -> Tuple[Union[List[str], List[SimpleITK.Image], List[numpy.ndarray]], List[models.Model]]:
    """
    Execute the MOOSE 3.0 image segmentation process.

    :param input_data: The input data to process, which can be one of the following:
                       - str: A file path to a NIfTI file.
                       - tuple[numpy.ndarray, tuple[float, float, float]]: A tuple containing a numpy array and spacing.
                       - SimpleITK.Image: An image object to process.
                       
    :param model_names: The name(s) of the model(s) to be used for segmentation.
    :type model_names: str or list[str]

    :param output_dir: Path to the directory where the output will be saved if the input is a file path.
    :type output_dir: Optional[str]

    :param accelerator: Specifies the accelerator type, e.g., "cpu" or "cuda".
    :type accelerator: Optional[str]

    :return: The output type aligns with the input type:
             - str (file path): If `input_data` is a file path.
             - SimpleITK.Image: If `input_data` is a SimpleITK.Image.
             - numpy.ndarray: If `input_data` is a numpy array.
    :rtype: str or SimpleITK.Image or numpy.ndarray

    :Example:
    >>> moose('/path/to/input/file', '[list, of, models]', '/path/to/output', 'cuda')
    >>> moose((numpy_array, (1.5, 1.5, 1.5)), 'model_name', '/path/to/output', 'cuda')
    >>> moose(simple_itk_image, 'model_name', '/path/to/output', 'cuda')
    """
    # Load the image and set a default filename based on input type
    if isinstance(input_data, str):
        image_raw = image_processing.image_read(input_data)
        image_raw_orientation_code = image_processing.image_get_orientation_code(image_raw)
        image = image_processing.image_reorient(image_raw, "RAS")
        file_name = file_utilities.get_nifti_file_stem(input_data)
    elif isinstance(input_data, SimpleITK.Image):
        image = input_data
        file_name = 'image_from_simpleitk'
    elif isinstance(input_data, tuple) and isinstance(input_data[0], numpy.ndarray) and isinstance(input_data[1], tuple):
        numpy_array, spacing = input_data
        image = SimpleITK.GetImageFromArray(numpy_array)
        image.SetSpacing(spacing)
        file_name = 'image_from_array'
    else:
        raise ValueError("Invalid input format. `input_data` must be either a file path (str), "
                         "a SimpleITK.Image, or a tuple (numpy array, spacing).")

    if isinstance(model_names, str):
        model_names = [model_names]

    output_manager = system.OutputManager(False, False)

    add_custom_trainers_to_local_nnunetv2()
    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_workflows = models.construct_model_workflows(model_names, output_manager)

    if accelerator is None:
        accelerator, _ = system.check_device(output_manager)

    subjects_information = ("temp", 0, 1)
    performance_observer = PerformanceObserver("temp", ', '.join(model_names))

    generated_segmentations = []
    used_models = []

    for segmentation_image, model in run_workflows(image, model_workflows, output_manager, performance_observer, accelerator, subjects_information):
        image_output = None
        if isinstance(input_data, str):
            if output_dir is None:
                output_dir = os.path.dirname(input_data)
            image_output = os.path.join(output_dir, f"{model.multilabel_prefix}segmentation_{file_name}.nii.gz")
            segmentation_image = image_processing.image_reorient(segmentation_image, image_raw_orientation_code)
            SimpleITK.WriteImage(segmentation_image, image_output)
        elif isinstance(input_data, SimpleITK.Image):
            image_output = segmentation_image
        elif isinstance(input_data, tuple):
            image_output = SimpleITK.GetArrayFromImage(segmentation_image)

        generated_segmentations.append(image_output)
        used_models.append(model)

    return generated_segmentations, used_models


def moose_subject(subject: str, subject_index: int, number_of_subjects: int, model_workflows: List[models.ModelWorkflow], accelerator: str,
                  output_manager: Union[system.OutputManager, None], benchmark: bool = False, single_labels: bool = False):
    subject_name = os.path.basename(subject)
    moose_information = system.get_system_information()
    moose_information["single_labels"] = single_labels
    moose_information["models"] = {}

    if output_manager is None:
        output_manager = system.OutputManager(False, False)

    output_manager.log_update(' ')
    output_manager.log_update(f' SUBJECT: {subject_name}')

    model_names = [model_workflow.target_model.model_identifier for model_workflow in model_workflows]

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

    start_time = time.time()
    output_manager.log_update(' ')
    output_manager.log_update(' RUNNING PREDICTION:')
    output_manager.log_update(' ')

    performance_observer.record_phase("Loading Image")
    CT_file_path = file_utilities.get_modality_file(subject, 'CT_')
    CT_file_name = file_utilities.get_nifti_file_stem(CT_file_path)
    CT_image = image_processing.image_read(CT_file_path)
    CT_image_orientation_code = image_processing.image_get_orientation_code(CT_image)
    CT_image_RAS = image_processing.image_reorient(CT_image, "RAS")

    subjects_information = (subject_name, subject_index, number_of_subjects)

    for segmentation_image, model in run_workflows(CT_image_RAS, model_workflows, output_manager, performance_observer, accelerator, subjects_information):
        model_information = {"model_directory": model.directory}
        segmentation_image = image_processing.image_reorient(segmentation_image, CT_image_orientation_code)
        performance_observer.record_phase("Writing Images and Statistics")
        segmentation_image_path = os.path.join(segmentations_dir, f"{model.multilabel_prefix}segmentation_{CT_file_name}.nii.gz")
        output_manager.log_update(f'     - Writing segmentation for {model}')
        model_information["segmentation_file"] = segmentation_image_path
        SimpleITK.WriteImage(segmentation_image, segmentation_image_path)
        output_manager.log_update(f'     - Writing organ indices for {model}')
        model.organ_indices_to_json(segmentations_dir)

        if single_labels:
            output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] extracting individual labels {subject_name}! {constants.ANSI_RESET}')
            output_manager.log_update(f'     - Extracting individual labels and writing them for {model}')
            individual_segmentations_dir = os.path.join(segmentations_dir, "individual_segmentations")
            image_processing.extract_labels_and_write(segmentation_image, model, individual_segmentations_dir)
            output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] labels extracted and written {subject_name}! {constants.ANSI_RESET}')

        # -----------------------------------------------
        # EXTRACT VOLUME STATISTICS AND HOUNSFIELD UNITS
        # -----------------------------------------------
        output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Extracting CT volume statistics for {subject_name} ({model})...')
        output_manager.log_update(f'     - Extracting volume statistics for {model}')
        out_vol_stats_csv = os.path.join(stats_dir, model.multilabel_prefix + subject_name + '_ct_volume.csv')
        image_processing.get_shape_statistics(segmentation_image, model, out_vol_stats_csv)
        output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] CT volume extracted for {subject_name}! {constants.ANSI_RESET}')
        model_information["VOL_stats_file"] = out_vol_stats_csv
        time.sleep(1)

        output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Extracting CT hounsfield statistics for {subject_name} ({model})...')
        output_manager.log_update(f'     - Extracting hounsfield statistics for {model}')
        out_hu_stats_csv = os.path.join(stats_dir, model.multilabel_prefix + subject_name + '_ct_hu_values.csv')
        image_processing.get_intensity_statistics(CT_image, segmentation_image, model, out_hu_stats_csv)
        output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] CT hounsfield statistics extracted for {subject_name}! {constants.ANSI_RESET}')
        model_information["HU_stats_file"] = out_hu_stats_csv
        time.sleep(1)

        # ----------------------------------
        # EXTRACT PET ACTIVITY
        # ----------------------------------
        PT_file_path = file_utilities.get_modality_file(subject, 'PT_')
        if PT_file_path is not None:
            PT_image = SimpleITK.ReadImage(PT_file_path)
            output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Extracting PET activity for {subject_name} ({model})...')
            output_manager.log_update(f'     - Extracting PET statistics for {model}')
            resampled_multilabel_image = image_processing.ImageResampler.reslice_identity(PT_image, segmentation_image, is_label_image=True)
            out_csv = os.path.join(stats_dir, model.multilabel_prefix + subject_name + '_pet_activity.csv')
            image_processing.get_intensity_statistics(PT_image, resampled_multilabel_image, model, out_csv)
            output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] PET activity extracted for {subject_name}! {constants.ANSI_RESET}')
            model_information["PET_stats_file"] = out_csv
            time.sleep(1)

        performance_observer.time_phase()
        moose_information["models"][model.model_identifier] = model_information

    system.write_information_json(moose_information, moose_dir)
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


def run_workflows(image: SimpleITK.Image, model_workflows: List[models.ModelWorkflow], output_manager: system.OutputManager, performance_observer: PerformanceObserver, accelerator: str, subjects_information: Tuple[str, int, int]) -> Iterator[Tuple[SimpleITK.Image, models.Model]]:
    current_image_array = SimpleITK.GetArrayFromImage(image)
    current_image_array_spacing = image.GetSpacing()[::-1]

    performance_observer.metadata_image_size = image.GetSize()
    performance_observer.time_phase()

    subject_name, subject_index, number_of_subjects = subjects_information

    for model_workflow in model_workflows:
        desired_spacing = model_workflow.initial_desired_spacing

        if current_image_array_spacing != desired_spacing:
            performance_observer.record_phase(f"Resampling Image: {'x'.join(map(str, desired_spacing))}")
            resampling_time_start = time.time()
            current_image_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)
            current_image_array_spacing = desired_spacing
            output_manager.log_update(f' - Resampling at {"x".join(map(str, desired_spacing))} took: {round((time.time() - resampling_time_start), 2)}s')
            performance_observer.time_phase()

        performance_observer.record_phase(f"Predicting: {model_workflow.target_model}")
        model_time_start = time.time()
        output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Running prediction for {subject_name} using {model_workflow[0]}...')
        output_manager.log_update(f'   - Model {model_workflow.target_model}')
        segmentation_array = predict.predict_from_array_by_iterator(current_image_array, model_workflow[0], accelerator, output_manager)

        if len(model_workflow) == 2:
            inference_fov_intensities = model_workflow[1].limit_fov["inference_fov_intensities"]
            if isinstance(inference_fov_intensities, int):
                inference_fov_intensities = [inference_fov_intensities]

            existing_intensities = numpy.unique(segmentation_array)
            if not all([intensity in existing_intensities for intensity in inference_fov_intensities]):
                output_manager.spinner_warn(f'[{subject_index + 1}/{number_of_subjects}] {subject_name}: organ to crop from not in initial FOV. No segmentation result ({model_workflow.target_model}) for this subject.')
                output_manager.log_update("     - Organ to crop from not in initial FOV.")
                performance_observer.time_phase()
                continue

            segmentation_array, desired_spacing = predict.cropped_fov_prediction_pipeline(image, segmentation_array, model_workflow, accelerator, output_manager)

        output_manager.log_update(f"     - Prediction complete for {model_workflow.target_model} within {round((time.time() - model_time_start) / 60, 1)} min.")
        performance_observer.time_phase()

        segmentation = SimpleITK.GetImageFromArray(segmentation_array)
        segmentation.SetSpacing(desired_spacing[::-1])
        segmentation.SetOrigin(image.GetOrigin())
        segmentation.SetDirection(image.GetDirection())
        resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

        yield resampled_segmentation, model_workflow.target_model


if __name__ == '__main__':
    main()
