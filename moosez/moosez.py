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
import sys
import argparse
import time
import SimpleITK
import colorama
import emoji
import numpy
import pandas as pd
import multiprocessing as mp
import concurrent.futures
from typing import Union, Tuple, List
from moosez import constants
from moosez import download
from moosez import file_utilities
from moosez import image_conversion
from moosez import input_validation
from moosez import system
from moosez import models
from moosez import streaming
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
        default=[constants.DEFAULT_MODEL],
        metavar="<MODEL_NAMES>",
        help=f"Choose the models for segmentation (default: {constants.DEFAULT_MODEL}) from the following:\n" +
             "\n".join(models.AVAILABLE_MODELS)
    )

    # Flat input: image files and / or folders of image files, no subject folders
    parser.add_argument(
        "-i", "--input",
        nargs='+',
        type=str,
        metavar="<INPUT>",
        help="Image files and/or folders of image files to segment, instead of -d. Needs -o."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        metavar="<OUTPUT_DIRECTORY>",
        help="Output folder for -i."
    )

    parser.add_argument(
        "-ns", "--no_statistics",
        action="store_true",
        default=False,
        help="Skip the volume and intensity statistics."
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
        const=0,
        type=int,
        help='Use several GPUs: one process per GPU, each with its share of the subjects (default: every GPU).'
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
        '-mi', '--model_install',
        type=str,
        metavar="<ARCHIVE.zip>",
        help='Install a model from a zip archive that holds its model folder (such as the weights of clin_ct_anatomy), '
             'without running segmentation.'
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

    if args.model_install:
        installed = models.install_model_archive(os.path.abspath(args.model_install), output_manager)
        if not installed:
            output_manager.console_update(f"{constants.ANSI_RED} {args.model_install} holds the model folder of no MOOSE "
                                          f"model.{constants.ANSI_RESET}")
            sys.exit(1)
        for identifier in installed:
            output_manager.console_update(f"{constants.ANSI_GREEN} Installed {identifier}. Run it with: moosez -i <images> "
                                          f"-o <output folder> -m {identifier}{constants.ANSI_RESET}")
        return

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

    model_names = list(dict.fromkeys(args.model_names))      # a model named twice runs once

    if args.input:
        if args.main_directory or not args.output:
            parser.error("-i needs -o and cannot be combined with -d")
        moose_files(args.input, os.path.abspath(args.output), model_names, output_manager, args.benchmark,
                    args.single_labels, not args.no_statistics)
        return

    if not args.main_directory:
        parser.error("Give the subject folders with -d, or image files / folders with -i and -o")

    parent_folder = os.path.abspath(args.main_directory)
    benchmark = args.benchmark
    single_labels = args.single_labels

    output_manager.configure_logging(parent_folder)
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update(f'                                     STARTING MOOSE-Z v{system.MOOSE_VERSION}                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')

    # ----------------------------------
    # DOWNLOADING THE MODEL
    # ----------------------------------

    requested_models = prepare_models(model_names, output_manager)

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    output_manager.log_update(f' ')
    output_manager.log_update(f'- Main directory: {parent_folder}')
    output_manager.log_update(f' ')
    output_manager.console_update(f' ')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    output_manager.console_update(f' ')

    modalities = input_validation.determine_model_expectations(requested_models, output_manager)
    accelerator, device_count = system.check_device(output_manager)

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

    num_moose_compliant_subjects = len(moose_compliant_subjects)

    if num_moose_compliant_subjects < 1:
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

    performance_observer = PerformanceObserver(f'All {num_moose_compliant_subjects} subjects', ', '.join(model_names))
    if benchmark:
        performance_observer.on()
    moose_directories = {subject: file_utilities.moose_folder_structure(subject) for subject in moose_compliant_subjects}
    herd = 1 if args.moose_herd is None else min(args.moose_herd or device_count or 1, device_count or 1,
                                                 num_moose_compliant_subjects)
    if herd > 1:
        # one process per GPU, each with every herd-th subject
        output_manager.spinner_update(f' Predicting on {herd} GPUs')
        with concurrent.futures.ProcessPoolExecutor(herd, mp_context=mp.get_context('spawn')) as executor:
            shares = [executor.submit(streaming.run_subjects, moose_compliant_subjects[gpu::herd], model_names,
                                      moose_directories, f"{accelerator}:{gpu}", None, single_labels,
                                      not args.no_statistics) for gpu in range(herd)]
        for share in shares:        # a process that failed fails the run, once the others are done
            share.result()
    else:
        streaming.run_subjects(moose_compliant_subjects, requested_models, moose_directories, accelerator,
                               output_manager, single_labels, not args.no_statistics)
    performance_observer.record_phase("Total Processing Done")
    if benchmark:
        performance_observer.off()

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
        csv_file_path = write_peak_performance(performance_observer, parent_folder)
        output_manager.log_update(f'  - Resource utilization written to {csv_file_path}')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update(f'                                     FINISHED MOOSE-Z v{system.MOOSE_VERSION}                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')


def write_peak_performance(performance_observer: PerformanceObserver, directory: str) -> str:
    """Writes the peak resources of a benchmarked run (-b) as a csv file into directory and returns its path."""
    df = pd.DataFrame([performance_observer.get_peak_resources()], columns=['Image', 'Model', 'Image Size', 'Runtime [s]', 'Peak Memory [GB]'])
    csv_file_path = os.path.join(directory, f'moosez-v{system.MOOSE_VERSION}_peak_performance_parameters.csv')
    df.to_csv(csv_file_path, index=False)
    return csv_file_path


def prepare_models(model_names: List[str], output_manager: system.OutputManager) -> List[models.Model]:
    """
    The requested models, and the models that give their roi, downloaded where needed before any prediction runs: the
    download progress does not run under the prediction spinner or into the prediction time, and -herd processes find
    every model in place. A model whose weights are missing ends the run with its message.
    """
    output_manager.console_update(f'')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    output_manager.console_update(f'')
    file_utilities.create_directory(system.MODELS_DIRECTORY_PATH)
    try:
        requested_models = [models.Model(model_name, output_manager) for model_name in model_names]
        models.with_roi_models(requested_models, output_manager)
    except FileNotFoundError as e:
        output_manager.console_update(f"{constants.ANSI_RED}{e}{constants.ANSI_RESET}")
        sys.exit(1)
    return requested_models


def moose_files(inputs: List[str], output_directory: str, model_names: List[str], output_manager: system.OutputManager,
                benchmark: bool = False, single_labels: bool = False, compute_statistics: bool = True) -> None:
    """
    moosez -i ... -o ...: image files and folders of image files (no subject folders); every image is given to every
    model, and the outputs go flat into output_directory (streaming.run_files).
    """
    image_files = streaming.collect_image_files(inputs)
    if not image_files:
        output_manager.console_update(f"{constants.ANSI_RED} No image files ({', '.join(streaming.IMAGE_FILE_ENDINGS)}) found in {', '.join(inputs)}.{constants.ANSI_RESET}")
        return

    os.makedirs(output_directory, exist_ok=True)
    output_manager.configure_logging(output_directory)
    output_manager.log_update(f'                                     STARTING MOOSE-Z v{system.MOOSE_VERSION}                                       ')
    output_manager.log_update(f'- {len(image_files)} images -> {output_directory}')
    requested_models = prepare_models(model_names, output_manager)
    output_manager.console_update(f'')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":crystal_ball:")} PREDICT:{constants.ANSI_RESET}')
    output_manager.console_update(f'')
    accelerator, _ = system.check_device(output_manager)

    information = system.get_system_information()
    information["single_labels"] = single_labels
    performance_observer = PerformanceObserver(f'{len(image_files)} images', ', '.join(model_names))
    if benchmark:
        performance_observer.on()
    start = time.time()
    output_manager.spinner_start(' Initiating')
    information["models"] = streaming.run_files(image_files, output_directory, requested_models, accelerator,
                                                output_manager, single_labels, compute_statistics)
    system.write_information_json(information, output_directory)
    elapsed = time.time() - start
    output_manager.spinner_succeed(f'{constants.ANSI_GREEN} All predictions done! | {len(image_files)} images, '
                                   f'{len(model_names)} models: {round(elapsed / 60, 1)} min | '
                                   f'per image: {round(elapsed / len(image_files), 1)} s {constants.ANSI_RESET}')
    output_manager.log_update(f' ALL IMAGES PROCESSED in {round(elapsed, 1)} s')
    if benchmark:
        performance_observer.record_phase("Total Processing Done")
        performance_observer.off()
        write_peak_performance(performance_observer, output_directory)


def moose(input_data: Union[str, Tuple[numpy.ndarray, Tuple[float, float, float]], SimpleITK.Image],
          model_names: Union[str, List[str]] = constants.DEFAULT_MODEL, output_dir: str = None, accelerator: str = None) -> Tuple[Union[List[str], List[SimpleITK.Image], List[numpy.ndarray]], List[models.Model]]:
    """
    Execute the MOOSE 3.0 image segmentation process.

    :param input_data: The input data to process, which can be one of the following:
                       - str: The file path to the NIfTI file.
                       - tuple[numpy.ndarray, tuple[float, float, float]]: A tuple containing a numpy array and spacing.
                       - SimpleITK.Image: An image object to process.
                       
    :param model_names: The name(s) of the model(s) to be used for segmentation (default: the default model).
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
    if isinstance(input_data, str):
        image = input_data
        file_name = file_utilities.get_nifti_file_stem(input_data)
        if output_dir is None:
            output_dir = os.path.dirname(input_data)
    elif isinstance(input_data, SimpleITK.Image):
        image = input_data
    elif isinstance(input_data, tuple) and isinstance(input_data[0], numpy.ndarray) and isinstance(input_data[1], tuple):
        # an array has no orientation: it is taken as RAS
        numpy_array, spacing = input_data
        image = SimpleITK.GetImageFromArray(numpy_array)
        image.SetSpacing(spacing)
        image.SetDirection(SimpleITK.DICOMOrientImageFilter_GetDirectionCosinesFromOrientation("RAS"))
    else:
        raise ValueError("Invalid input format. `input_data` must be either a file path (str), "
                         "a SimpleITK.Image, or a tuple (numpy array, spacing).")

    if isinstance(model_names, str):
        model_names = [model_names]

    output_manager = system.OutputManager(False, False)
    with output_manager.manage_nnUNet_output(), \
            streaming.MoosePredictor(model_names, accelerator, output_manager=output_manager) as predictor:
        # a file gives files, an image gives images
        outputs = {model.model_identifier: {'file': streaming.segmentation_file(model, output_dir, file_name)}
                   for model in predictor.requested} if isinstance(input_data, str) else {}
        results = predictor.predict([image], [outputs])[0]
    # a model whose roi is not in the image gives no result
    used_models = [model for model in predictor.requested if results.get(model.model_identifier) is not None]
    segmentations = [results[model.model_identifier] for model in used_models]
    if isinstance(input_data, tuple):       # an array gives arrays
        segmentations = [SimpleITK.GetArrayFromImage(segmentation) for segmentation in segmentations]
    return segmentations, used_models


if __name__ == '__main__':
    main()
