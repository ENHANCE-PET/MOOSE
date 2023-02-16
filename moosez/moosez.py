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

    logging.info('****************************************************************************************************')
    logging.info('                                     STARTING MOOSE-Z V.2.0.0                                       ')
    logging.info('****************************************************************************************************')

    # INPUT VALIDATION AND PREPARATION

    logging.info(' ')
    logging.info('- Main directory: ' + parent_folder)
    logging.info('- Model name: ' + model_name)
    logging.info(' ')
    print(' ')
    print(f'{constants.ANSI_VIOLET} NOTE:{constants.ANSI_RESET}')
    print(' ')
    display.expectations(model_name)

    # DOWNLOADING THE MODEL
    print('')
    print(f'{constants.ANSI_VIOLET} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    print('')
    model_path = file_utilities.nnunet_folder_structure()
    download.model(model_name, model_path)

    # CHECKING FOR EXPECTED MODALITIES

    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        os.dup2(devnull.fileno(), 1)

        # Call the function with suppressed output
        modalities = display.expected_modality(model_name)

        # Restore stdout
        os.dup2(old_stdout, 1)

    # INPUT STANDARDIZATION

    print('')
    print(f'{constants.ANSI_VIOLET} STANDARDIZING INPUT DATA:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' STANDARDIZING INPUT DATA:')
    logging.info(' ')
    image_conversion.standardize_to_nifti(parent_folder)
    print(f"{constants.ANSI_GREEN} Standardization complete.{constants.ANSI_RESET}")
    logging.info(" Standardization complete.")

    # SETTING UP DIRECTORY STRUCTURE

    print('')
    print(f'{constants.ANSI_VIOLET} SETTING UP MOOSE-Z DIRECTORY FOR BATCH PROCESSING:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' SETTING UP MOOSE-Z DIRECTORY FOR BATCH PROCESSING:')
    logging.info(' ')
    moose_dir, input_dirs, output_dirs = file_utilities.moose_folder_structure(parent_folder, model_name, modalities)
    print(f" MOOSE directory for the current run set at: {moose_dir}")
    logging.info(f" MOOSE directory for the current run set at: {moose_dir}")

    # PREPARE THE DATA FOR PREDICTION

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if
                os.path.isdir(os.path.join(parent_folder, d))]
    if moose_dir in subjects:
        subjects.remove(moose_dir)
    moose_compliant_subjects = input_validation.select_moose_compliant_subjects(subjects, modalities)
    file_utilities.organise_files_by_modality(moose_compliant_subjects, modalities, moose_dir)
    spinner = Halo(text='Preparing data for prediction', spinner='dots')
    spinner.start()
    for input_dir in input_dirs:
        input_validation.make_nnunet_compatible(input_dir)
    spinner.succeed('Data ready for prediction.')

    # RUN PREDICTION

    print('')
    print(f'{constants.ANSI_VIOLET} RUNNING PREDICTION:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' RUNNING PREDICTION:')
    logging.info(' ')
    # predict.run_prediction(model_name, input_dirs, output_dirs)
    # Run the segmentation on the standardized data and save the results in the output directory


# predict.run_prediction(model_name, input_dirs, output_dirs)
# Push back the results to their original locations


if __name__ == '__main__':
    main()
