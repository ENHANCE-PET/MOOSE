#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
#         Sebastian Gutschmayer
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# This module handles image conversion for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform image conversion.
#
# ----------------------------------------------------------------------------------------------------------------------

import logging
import re
import subprocess

from halo import Halo
import SimpleITK


def dcm2nii(dicom_dir: str) -> None:
    """
    Convert DICOM images to NIFTI using dcm2niix

    :param dicom_dir: str, Directory containing the DICOM images
    """
    cmd = f"dcm2niix -f %b {re.escape(dicom_dir)}"
    logging.info(f"Converting DICOM images in {dicom_dir} to NIFTI")
    spinner = Halo(text=f"Converting DICOM images in {dicom_dir} to NIFTI", spinner='dots')
    spinner.start()
    subprocess.run(cmd, shell=True, capture_output=True)
    spinner.succeed(text=f"Converted DICOM images in {dicom_dir} to NIFTI")
    logging.info("Conversion completed successfully")

    
def read_dicom_folder(folder_path: str) -> SimpleITK.Image:
    """
    Reads a folder of DICOM files and returns the image

    :param folder_path: str, Directory to get DICOM files from
    """
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)

    dicom_image = reader.Execute()
    return dicom_image


def anyformat2nii(input_path: str, output_directory: str = None) -> None:
    """
    Converts any image format known to ITK to NIFTI

    :param input_path: str, Directory OR filename to convert to nii.gz
    :param output_directory: str, optional output directory to write the image to.
    """
    if os.path.isdir(input_path):
        image_probe = os.listdir(input_path)[0]
        if image_probe.endswith('IMA') or image_probe.endswith('dcm'):
            # Get a few infos from the probed file's basename
            image_probe_basename = os.path.basename(image_probe)
            input_extension = image_probe_basename[image_probe_basename.rfind('.') + 1:]
            image_probe_stem = os.path.basename(input_path)
            logging.info(f"Converting {input_extension} image to NIFTI")

            # Reading the dicom directory
            output_image = read_dicom_folder(input_path)

            # The new image stem is generated here
            output_image_basename = image_probe_stem + '.nii.gz'
        else:
            logging.info(f"Conversion ERROR: The specified directory does not contain a dicom or IMA file.")
            return

    elif os.path.isfile(input_path):
        if input_path.endswith('.nii.gz'):
            logging.info(f"A compressed NIFTI file was provided. No conversion will follow.")
        else:
            # Get a few infos from the file basename
            input_image_basename = os.path.basename(input_path)
            input_extension = input_image_basename[input_image_basename.rfind('.') + 1:]
            input_image_stem = input_image_basename[:input_image_basename.rfind('.')]
            logging.info(f"Converting {input_extension} image to NIFTI")

            # Determining if the file is a DICOM file or not, so we can search deeper
            if input_extension == 'IMA' or input_extension == 'dcm':
                input_directory = os.path.dirname(input_path)
                logging.info(f'A {input_extension} file was provided. Scanning {input_directory} to find rest of series.')
                output_image = read_dicom_folder(input_directory)
            else:
                output_image = SimpleITK.ReadImage(input_path)

            # The new image stem is generated here
            output_image_basename = input_image_stem + '.nii.gz'
    else:
        logging.info(f"Conversion ERROR: Neither a valid directory nor file path was specified.")
        return

    # The output path is generated here
    if output_directory is None:
        output_directory = os.path.dirname(input_path)
    output_image_path = os.path.join(output_directory, output_image_basename)

    # The image is written here
    spinner = Halo(text=f"Converting {input_extension} image to NIFTI", spinner='dots')
    spinner.start()
    SimpleITK.WriteImage(output_image, output_image_path)
    spinner.succeed(text=f"Converted {input_extension} image to NIFTI")
    logging.info("Conversion completed successfully")
