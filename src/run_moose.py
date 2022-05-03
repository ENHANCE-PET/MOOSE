#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ***********************************************************************************************************************
# File: run_moose.py
# Project: MOOSE Version 1.0
# Created: 21.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: Lalith.Shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This corresponds to the main file for running MOOSE.
# License: Apache 2.0
# **********************************************************************************************************************

import argparse
import logging
import os
import pathlib
import timeit
import openpyxl
import checkArgs
import constants as c
import fileOp as fop
import imageIO
import imageOp as iop
import inferenceEngine as ie
import postprocess as pp
import errorAnalysis as ea
from datetime import datetime

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                    filename=datetime.now().strftime('moose-%H-%M-%d-%m-%Y.log'),
                    filemode='w')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--main_folder", type=str, help="Main directory that contain the patient folders, "
                                                              "each folder in turn should contain PET/CT data",
                        required=True)
    parser.add_argument("-ea", "--error_analysis", type=bool, help="If set to True, the error analysis will be "
                                                                   "performed", default=True)

    args = parser.parse_args()

    main_folder = args.main_folder
    if not checkArgs.dir_exists(main_folder):
        logging.error("The main folder does not exist")
        print("The main folder does not exist")
        exit(1)

    error_analysis = args.error_analysis

    logging.info('****************************************************************************************************')
    logging.info('                                     STARTING MOOSE V.1.0                                           ')
    logging.info('****************************************************************************************************')
    start = timeit.default_timer()
    fop.display_logo()
    logging.info(' ')
    logging.info('INPUT ARGUMENTS')
    logging.info('------------------')
    logging.info('- Main folder: ' + main_folder)
    logging.info('- Error analysis: ' + str(args.error_analysis) + '[1 = True, 0 = False]')
    logging.info('- Total number of subjects to MOOSE: ' + str(len(fop.get_folders(main_folder))))
    logging.info(' ')

    # --------------------------------------Processing individual subjects------------------------------------------

    subject_folders = fop.get_folders(main_folder)

    for subject_folder in subject_folders:
        logging.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        logging.info(f"- Working folder: {subject_folder}")
        processing_folder = str(pathlib.Path(subject_folder).stem)
        logging.info(f"- Processing subject:  {processing_folder}")
        sub_folders = fop.get_folders(subject_folder)
        logging.info(f"- Number of folders in subject {processing_folder}: {str(len(sub_folders))}")
        logging.info(f"- Folder names: {sub_folders}")
        logging.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        logging.info(' ')
        imageIO.dcm2nii(subject_folder)
        dicom_jsons = fop.get_files(subject_folder, '*json')
        if len(dicom_jsons) == 0:
            logging.error(f'Provided data possibly not in DICOM format in folder {subject_folder}')
            print(f'Provided data possibly not in DICOM format in folder {subject_folder}')
            exit(1)
        fop.organise_nii_files_in_folders(dir_path=subject_folder, json_files=dicom_jsons)
        logging.info('Looking for modality type and organising the files accordingly')
        if checkArgs.dir_exists(os.path.join(subject_folder, 'PT')):
            pet = True
        else:
            pet = False
        if checkArgs.dir_exists(os.path.join(subject_folder, 'CT')):
            ct = True
        else:
            ct = False

        if pet and not ct:
            logging.error('Only PET data found in folder ' + subject_folder + ', MOOSE cannot proceed only with PET '
                                                                              'data')
            print('Only PET data found in folder ' + subject_folder + ', MOOSE cannot proceed only with PET '
                                                                      'data')
            exit(1)
        elif ct and not pet:
            logging.info('Only CT data found in folder ' + subject_folder + ', MOOSE will construct non-cerebral '
                                                                            'tissue atlas (n=37) based on CT data')
            print('Only CT data found in folder ' + subject_folder + ', MOOSE will construct non-cerebral '
                                                                     'tissue atlas (n=37) based on CT data')
            logging.info('--------------------------------------------------------------------------------------------')
            logging.info('                       Initiating CT segmentation protocols                                 ')
            logging.info('--------------------------------------------------------------------------------------------')
            print('Initiating CT segmentation protocols')
            out_dir = fop.make_dir(subject_folder, 'labels')
            sim_space_dir = fop.make_dir(out_dir, 'sim_space')
            temp_dir = fop.make_dir(subject_folder, 'temp')
            ct_file = fop.get_files(os.path.join(subject_folder, 'CT'), '*nii')
            ct_atlas = ie.segment_ct(ct_file[0], out_dir)
            logging.info('CT segmentation completed')
            ea.similarity_space(ct_atlas, sim_space_dir, os.path.join(subject_folder,
                                                                      processing_folder
                                                                      + '-Risk-of-segmentation-error.csv'))
        elif pet and ct:
            logging.info('Both PET and CT data found in folder ' + subject_folder + ', MOOSE will construct the '
                                                                                    'full tissue atlas (n=120) '
                                                                                    'based on both PET and CT data')
            print('Both PET and CT data found in folder ' + subject_folder + ', MOOSE will construct the full '
                                                                             'tissue atlas (n=120) based on both '
                                                                             'PET and CT data')
            logging.info('Initiating PET/CT segmentation protocols')
            print('Initiating PET/CT segmentation protocols')
            out_dir = fop.make_dir(subject_folder, 'labels')
            temp_dir = fop.make_dir(subject_folder, 'temp')
            sim_space_dir = fop.make_dir(out_dir, 'sim_space')
            logging.info('Output folder: ' + out_dir)
            print('Output folder: ' + out_dir)
            ct_file = fop.get_files(os.path.join(subject_folder, 'CT'), '*nii')
            moose_ct_atlas = ie.segment_ct(ct_file[0], out_dir)
            logging.info('Aligning PET and CT data using diffeomorphic registration')
            print('Aligning PET and CT data using diffeomorphic registration')
            pet_file = fop.get_files(os.path.join(subject_folder, 'PT'), '*nii')
            aligned_moose_ct_atlas = pp.align_pet_ct(pet_file[0], fop.get_files(os.path.join(subject_folder, 'CT'),
                                                                                '*nii.gz')[0],
                                                     moose_ct_atlas)
            logging.info('PET/CT alignment completed')
            print('PET/CT alignment completed')
            logging.info('Calculating SUV parameters for SUV extraction...')
            pet_stem = pathlib.Path(pet_file[0]).stem
            pet_json = fop.get_files(subject_folder, pet_stem + '*json')[0]
            pet_folder = imageIO.return_dicomdir_modality(sub_folders, 'PT')
            pet_dcm_files = fop.get_files(pet_folder, '*')
            suv_param = iop.get_suv_parameters(pet_dcm_files[round((len(pet_dcm_files) / 2))])
            logging.info('Converting PET image to SUV Image...')
            suv_image = iop.convert_bq_to_suv(bq_image=pet_file[0], out_suv_image=fop.add_prefix(pet_file[0],
                                                                                                 'SUV-'),
                                              suv_parameters=suv_param)
            if pp.brain_exists(moose_ct_atlas):
                logging.info('Brain found in field-of-view of PET/CT data...')
                print('Brain found in field-of-view of PET/CT data...')
                logging.info('Cropping brain from PET image using the aligned CT brain mask')
                print('Cropping brain from PET image using the aligned CT brain mask')
                cropped_pet_brain = iop.crop_image_using_mask(image_to_crop=pet_file[0],
                                                              multilabel_mask=aligned_moose_ct_atlas,
                                                              out_image=os.path.join(temp_dir,
                                                                                     c.CROPPED_BRAIN_FROM_PET),
                                                              label_intensity=4)
                logging.info('Brain region cropped from PET...')
                print('Brain region cropped from PET...')
                logging.info('Segmenting 83 tissue types of the Hammersmith atlas from the cropped PET brain')
                brain_seg = ie.segment_pt(cropped_pet_brain, out_dir)
                logging.info('83 tissue types segmented from the cropped PET brain')
                print('83 tissue types segmented from the cropped PET brain')
                logging.info('Merging PET and CT segmentations to construct the entire atlas...')
                print('Merging PET and CT segmentations to construct the entire atlas...')
                merged_seg = pp.merge_pet_ct_segmentations(brain_seg, aligned_moose_ct_atlas, os.path.join(out_dir,
                                                                                                           'MOOSE'
                                                                                                           '-unified'
                                                                                                           '-PET-CT'
                                                                                                           '-atlas'
                                                                                                           '.nii.gz'))
                logging.info(f'PET/CT segmentation completed and unified atlas constructed and stored in '
                             f'{os.path.join(out_dir, "MOOSE-unified-PET-CT-atlas.nii.gz")}')
                print(f'PET/CT segmentation completed and unified atlas constructed and stored in '
                      f'{os.path.join(out_dir, "MOOSE-unified-PET-CT-atlas.nii.gz")}')
                logging.info('Extracting SUV values from the PET image using the MOOSE atlas...')
                print('Extracting SUV values from the PET image using the MOOSE atlas...')
                iop.get_intensity_statistics(suv_image, merged_seg, os.path.join(subject_folder,
                                                                                 processing_folder +
                                                                                 '-SUV-values.csv'))
                logging.info('SUV values extracted from the PET image using the MOOSE atlas...')
                logging.info(
                    'SUV values stored in ' + os.path.join(subject_folder, processing_folder + '-SUV-values.csv'))

            else:
                logging.info('No brain found in field-of-view of PET/CT data...')
                print('No brain found in field-of-view of PET/CT data...')
                no_brain_seg = fop.add_prefix_rename(aligned_moose_ct_atlas, 'No-Brain-FOV-')
                logging.info('Writing out the aligned CT atlas without the brain...')
                print('Writing out the aligned CT atlas without the brain...')
                logging.info(f'MOOSE atlas stored in {no_brain_seg}')
                print(f'MOOSE atlas stored in {no_brain_seg}')
                logging.info('Extracting SUV values from the PET image using the MOOSE atlas...')
                print('Extracting SUV values from the PET image using the MOOSE atlas...')
                iop.get_intensity_statistics(suv_image, no_brain_seg, os.path.join(subject_folder,
                                                                                   processing_folder +
                                                                                   '-SUV-values.csv'))
                logging.info('SUV values extracted from the PET image using the MOOSE atlas...')
                logging.info(
                    'SUV values stored in ' + os.path.join(subject_folder, processing_folder + '-SUV-values.csv'))

            ea.similarity_space(moose_ct_atlas, sim_space_dir,
                                os.path.join(subject_folder, processing_folder + '-Risk-of-segmentation-error.csv'))

        else:
            logging.error('No PET or CT data found in folder ' + subject_folder + ', MOOSE cannot proceed, '
                                                                                  'please check data')
            print(
                'No PET or CT data found in folder ' + subject_folder + ', MOOSE cannot proceed, please check data')
            exit(1)
        moose_dir = os.path.join(subject_folder, 'MOOSE' + '-' + processing_folder)
        logging.info('Initiating cleanup of MOOSE output...')
        contents = fop.get_folders(subject_folder)
        contents_to_exclude = sub_folders
        contents_to_move = [x for x in contents if x not in contents_to_exclude]
        fop.move_contents(contents_to_move, moose_dir)
        logging.info(f'MOOSE output cleaned up...and can be found in {moose_dir}')
    stop = timeit.default_timer()
    logging.info(' ')
    logging.info(f"Total time taken for MOOSE processing: {(stop - start) / 60:.2f} minutes")
    logging.info(' ')
