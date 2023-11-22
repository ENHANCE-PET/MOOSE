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

import argparse
import glob
import logging
import os
import time
from datetime import datetime

import SimpleITK
import colorama
import emoji
from halo import Halo
from moosez import constants
from moosez import display
from moosez import download
from moosez import file_utilities
from moosez import image_conversion
from moosez import image_processing
from moosez import input_validation
from moosez import predict
from moosez import resources
from moosez.image_processing import ImageResampler
from moosez.nnUNet_custom_trainer.utility import add_custom_trainers_to_local_nnunetv2
from moosez.resources import MODELS, AVAILABLE_MODELS

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                    filename=datetime.now().strftime('moosez-v.2.0.0.%H-%M-%d-%m-%Y.log'),
                    filemode='w')


def main():
    colorama.init()

    # Argument parser
    parser = argparse.ArgumentParser(
        description=display.get_usage_message(),
        formatter_class=argparse.RawTextHelpFormatter,  # To retain the custom formatting
        add_help=False  # We'll add our own help option later
    )

    # Main directory containing subject folders
    parser.add_argument(
        "-d", "--main_directory",
        type=str,
        required=True,
        metavar="<MAIN_DIRECTORY>",
        help="Specify the main directory containing subject folders."
    )

    # Name of the model to use for segmentation
    model_help_text = "Choose the model for segmentation from the following:\n" + "\n".join(AVAILABLE_MODELS)
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        choices=AVAILABLE_MODELS,
        required=True,
        metavar="<MODEL_NAME>",
        help=model_help_text
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
    model_name = args.model_name

    display.logo()
    display.citation()

    logging.info('----------------------------------------------------------------------------------------------------')
    logging.info('                                     STARTING MOOSE-Z V.2.0.0                                       ')
    logging.info('----------------------------------------------------------------------------------------------------')

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    logging.info(' ')
    logging.info('- Main directory: ' + parent_folder)
    logging.info('- Model name: ' + model_name)
    logging.info(' ')
    print(' ')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    print(' ')
    modalities = display.expectations(model_name)
    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    logging.info('- Custom trainer: ' + custom_trainer_status)
    accelerator = resources.check_cuda()

    # ----------------------------------
    # DOWNLOADING THE MODEL
    # ----------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    print('')
    model_path = constants.NNUNET_RESULTS_FOLDER
    file_utilities.create_directory(model_path)
    download.model(model_name, model_path)

    # ----------------------------------
    # INPUT STANDARDIZATION
    # ----------------------------------

    print('')
    print(
        f'{constants.ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} STANDARDIZING INPUT DATA TO NIFTI:{constants.ANSI_RESET}')
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

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if
                os.path.isdir(os.path.join(parent_folder, d))]
    moose_compliant_subjects = input_validation.select_moose_compliant_subjects(subjects, modalities)

    num_subjects = len(moose_compliant_subjects)

    if num_subjects < 1:
        print(
            f'{constants.ANSI_RED} {emoji.emojize(":cross_mark:")} No moose compliant subject found to continue!{constants.ANSI_RESET} {emoji.emojize(":light_bulb:")} See: https://github.com/QIMP-Team/MOOSE#directory-structure-and-naming-conventions-for-moose-%EF%B8%8F')
        return

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR MOOSE COMPLIANT SUBJECTS
    # -------------------------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":crystal_ball:")} PREDICT:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' PERFORMING PREDICTION:')
    logging.info(' ')

    spinner = Halo(text=' Initiating', spinner='dots')
    spinner.start()
    start_total_time = time.time()

    for i, subject in enumerate(moose_compliant_subjects):
        # SETTING UP DIRECTORY STRUCTURE
        spinner.text = f'[{i + 1}/{num_subjects}] Setting up directory structure for {os.path.basename(subject)}...'
        logging.info(' ')
        logging.info(f'{constants.ANSI_VIOLET} SETTING UP MOOSE-Z DIRECTORY:'
                     f'{constants.ANSI_RESET}')
        logging.info(' ')
        moose_dir, input_dirs, output_dir, stats_dir = file_utilities.moose_folder_structure(subject, model_name,
                                                                                             modalities)
        logging.info(f" MOOSE directory for subject {os.path.basename(subject)} at: {moose_dir}")

        # ORGANISE DATA ACCORDING TO MODALITY
        spinner.text = f'[{i + 1}/{num_subjects}] Organising data according to modality for {os.path.basename(subject)}...'
        file_utilities.organise_files_by_modality([subject], modalities, moose_dir)

        # PREPARE THE DATA FOR PREDICTION
        spinner.text = f'[{i + 1}/{num_subjects}] Preparing data for prediction for {os.path.basename(subject)}...'
        for input_dir in input_dirs:
            input_validation.make_nnunet_compatible(input_dir)
        logging.info(f" {constants.ANSI_GREEN}Data preparation complete using {model_name} for subject "
                     f"{os.path.basename(subject)}{constants.ANSI_RESET}")

        # RUN PREDICTION
        start_time = time.time()
        logging.info(' ')
        logging.info(' RUNNING PREDICTION:')
        logging.info(' ')
        spinner.text = f'[{i + 1}/{num_subjects}] Running prediction for {os.path.basename(subject)} using {model_name}...'

        for input_dir in input_dirs:
            predict.predict(model_name, input_dir, output_dir, accelerator)
        logging.info(f"Prediction complete using {model_name}.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        spinner.text = f' {constants.ANSI_GREEN}[{i + 1}/{num_subjects}] Prediction done for {os.path.basename(subject)} using {model_name}!' \
                       f' | Elapsed time: {round(elapsed_time / 60, 1)} min{constants.ANSI_RESET}'
        time.sleep(3)
        logging.info(
            f' {constants.ANSI_GREEN}[{i + 1}/{num_subjects}] Prediction done for {os.path.basename(subject)} using {model_name}!' f' | Elapsed time: {round(elapsed_time / 60, 1)} min{constants.ANSI_RESET}')
        # ----------------------------------
        # EXTRACT PET ACTIVITY
        # ----------------------------------
        pet_file = file_utilities.find_pet_file(subject)
        if pet_file is not None:
            pet_image = SimpleITK.ReadImage(pet_file)
            spinner.text = f'[{i + 1}/{num_subjects}] Extracting PET activity for {os.path.basename(subject)}...'
            multilabel_file = glob.glob(os.path.join(output_dir, MODELS[model_name]["multilabel_prefix"] + '*nii*'))[0]
            multilabel_image = SimpleITK.ReadImage(multilabel_file)
            resampled_multilabel_image = ImageResampler.reslice_identity(reference_image=pet_image,
                                                                         moving_image=multilabel_image,
                                                                         is_label_image=True)
            out_csv = os.path.join(stats_dir, os.path.basename(subject) + '_pet_activity.csv')
            image_processing.get_intensity_statistics(pet_image, resampled_multilabel_image, model_name, out_csv)
            spinner.text = f'{constants.ANSI_GREEN} [{i + 1}/{num_subjects}] PET activity extracted for {os.path.basename(subject)}! ' \
                           f'{constants.ANSI_RESET}'
            time.sleep(3)

        # ----------------------------------
        # EXTRACT VOLUME STATISTICS
        # ----------------------------------
        spinner.text = f'[{i + 1}/{num_subjects}] Extracting CT volume statistics for {os.path.basename(subject)}...'
        multilabel_file = glob.glob(os.path.join(output_dir, MODELS[model_name]["multilabel_prefix"] + '*nii*'))[0]
        multilabel_image = SimpleITK.ReadImage(multilabel_file)
        out_csv = os.path.join(stats_dir, os.path.basename(subject) + '_ct_volume.csv')
        image_processing.get_shape_statistics(multilabel_image, model_name, out_csv)
        spinner.text = f'{constants.ANSI_GREEN} [{i + 1}/{num_subjects}] CT volume extracted for {os.path.basename(subject)}! ' \
                       f'{constants.ANSI_RESET}'
        time.sleep(1)

    end_total_time = time.time()
    total_elapsed_time = (end_total_time - start_total_time) / 60
    time_per_dataset = total_elapsed_time / len(moose_compliant_subjects)

    spinner.succeed(f'{constants.ANSI_GREEN} All predictions done! | Total elapsed time for '
                    f'{len(moose_compliant_subjects)} datasets: {round(total_elapsed_time, 1)} min'
                    f' | Time per dataset: {round(time_per_dataset, 2)} min {constants.ANSI_RESET}')


def moose(model_name: str, input_dir: str, output_dir: str, accelerator: str) -> None:
    """
    Execute the MOOSE 2.0 image segmentation process.

    This function carries out the following steps:
    1. Sets the path for model results.
    2. Creates the required directory for the model.
    3. Downloads the model based on the provided `model_name`.
    4. Validates and prepares the input directory to be compatible with nnUNet.
    5. Executes the prediction process.

    :param model_name: The name of the model to be used for predictions. This model will be downloaded and used 
                       for the image segmentation process.
    :type model_name: str

    :param input_dir: Path to the directory containing the images (in nifti, either .nii or .nii.gz) to be processed.
    :type input_dir: str

    :param output_dir: Path to the directory where the segmented output will be saved.
    :type output_dir: str

    :param accelerator: Specifies the type of accelerator to be used. Common values include "cpu" and "cuda" for 
                        GPU acceleration.
    :type accelerator: str

    :return: None
    :rtype: None

    :Example:

    >>> moose('clin_ct_organs', '/path/to/input/images', '/path/to/save/output', 'cuda')

    """
    model_path = constants.NNUNET_RESULTS_FOLDER
    file_utilities.create_directory(model_path)
    download.model(model_name, model_path)
    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    logging.info('- Custom trainer: ' + custom_trainer_status)
    input_validation.make_nnunet_compatible(input_dir)
    predict.predict(model_name, input_dir, output_dir, accelerator)


if __name__ == '__main__':
    main()
