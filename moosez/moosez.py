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
import logging
import os
from datetime import datetime
from halo import Halo
from moosez import display
from moosez import download
from moosez import file_utilities
from moosez import input_validation
from moosez import predict
from moosez import constants
from moosez import image_conversion
from threading import Thread
import tqdm

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                    filename=datetime.now().strftime('moosez-v.2.0.0.%H-%M-%d-%m-%Y.log'),
                    filemode='w')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--main_directory", type=str, help="Main directory containing subject folders",
                        required=True)

    parser.add_argument("-m", "--model_name", type=str, help="Name of the model to use for segmentation",
                        choices=["clin_ct_bones",
                                 "clin_ct_ribs",
                                 "clin_ct_vertebrae",
                                 "clin_ct_muscles",
                                 "clin_ct_lungs",
                                 "clin_ct_fat",
                                 "clin_ct_vessels",
                                 "clin_ct_organs",
                                 "clin_pt_fdg_tumor",
                                 "clin_ct_all",
                                 "clin_fdg_pt_ct_all",
                                 "preclin_mr_all"], required=True)
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
    print(f'{constants.ANSI_VIOLET} NOTE:{constants.ANSI_RESET}')
    print(' ')
    display.expectations(model_name)

    # ----------------------------------
    # DOWNLOADING THE MODEL
    # ----------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    print('')
    model_path = constants.NNUNET_RESULTS_FOLDER
    file_utilities.create_directory(model_path)
    download.model(model_name, model_path)

    # ----------------------------------
    # CHECKING FOR EXPECTED MODALITIES
    # ----------------------------------

    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        os.dup2(devnull.fileno(), 1)

        # Call the function with suppressed output
        modalities = display.expected_modality(model_name)

        # Restore stdout
        os.dup2(old_stdout, 1)

    # ----------------------------------
    # INPUT STANDARDIZATION
    # ----------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} STANDARDIZING INPUT DATA TO NIFTI:{constants.ANSI_RESET}')
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

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR MOOSE COMPLIANT SUBJECTS
    # -------------------------------------------------

    for subject in tqdm.tqdm(moose_compliant_subjects, desc=' Processing MOOSE-Z compliant directories'):

        # SETTING UP DIRECTORY STRUCTURE

        logging.info(' ')
        logging.info(f'{constants.ANSI_VIOLET} SETTING UP MOOSE-Z DIRECTORY:'
                     f'{constants.ANSI_RESET}')
        logging.info(' ')
        moose_dir, input_dirs, output_dir = file_utilities.moose_folder_structure(subject, model_name,
                                                                                  modalities)
        logging.info(f" MOOSE directory for subject {os.path.basename(subject)} at: {moose_dir}")

        # ORGANISE DATA ACCORDING TO MODALITY

        file_utilities.organise_files_by_modality([subject], modalities, moose_dir)

        # PREPARE THE DATA FOR PREDICTION

        for input_dir in input_dirs:
            input_validation.make_nnunet_compatible(input_dir)
        logging.info(f" {constants.ANSI_GREEN}Data preparation complete using {model_name} for subject "
                     f"{os.path.basename(subject)}{constants.ANSI_RESET}")

        # RUN PREDICTION

        logging.info(' ')
        logging.info(' RUNNING PREDICTION:')
        logging.info(' ')
        with tqdm.tqdm(total=len(input_dirs), desc=f"Running prediction using {model_name}", unit='input dir') as pbar:
            for input_dir, output_dir in zip(input_dirs, output_dir):
                predict.predict(model_name, input_dir, output_dir)
                pbar.update()
        logging.info(f"Prediction complete using {model_name}.")


# predict.run_prediction(model_name, input_dirs, output_dirs)
# Push back the results to their original locations


if __name__ == '__main__':
    main()
