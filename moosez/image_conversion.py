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

import contextlib
import io
import os
import re
import unicodedata

import SimpleITK
import dicom2nifti
import pydicom
from rich.progress import Progress


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


def non_nifti_to_nifti(input_path: str, output_directory: str = None) -> None:
    """
    Converts any image format known to ITK to NIFTI

    :param input_path: str, Directory OR filename to convert to nii.gz
    :param output_directory: str, optional output directory to write the image to.
    """

    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist.")
        return

    # Processing a directory
    if os.path.isdir(input_path):
        dicom_info = create_dicom_lookup(input_path)
        nifti_dir = dcm2niix(input_path)
        rename_nifti_files(nifti_dir, dicom_info)
        return

    # Processing a file
    if os.path.isfile(input_path):
        # Ignore hidden or already processed files
        _, filename = os.path.split(input_path)
        if filename.startswith('.') or filename.endswith(('.nii.gz', '.nii')):
            return
        else:
            output_image = SimpleITK.ReadImage(input_path)
            output_image_basename = f"{os.path.splitext(filename)[0]}.nii"

    if output_directory is None:
        output_directory = os.path.dirname(input_path)

    output_image_path = os.path.join(output_directory, output_image_basename)
    SimpleITK.WriteImage(output_image, output_image_path)


def standardize_to_nifti(parent_dir: str):
    """
    Converts all images in a parent directory to NIFTI
    """
    # go through the subdirectories
    subjects = os.listdir(parent_dir)
    # get only the directories
    subjects = [subject for subject in subjects if os.path.isdir(os.path.join(parent_dir, subject))]

    with Progress() as progress:
        task = progress.add_task("[white] Processing subjects...", total=len(subjects))
        for subject in subjects:
            subject_path = os.path.join(parent_dir, subject)
            if os.path.isdir(subject_path):
                images = os.listdir(subject_path)
                for image in images:
                    if os.path.isdir(os.path.join(subject_path, image)):
                        image_path = os.path.join(subject_path, image)
                        non_nifti_to_nifti(image_path)
                    elif os.path.isfile(os.path.join(subject_path, image)):
                        image_path = os.path.join(subject_path, image)
                        non_nifti_to_nifti(image_path)
            else:
                continue
            progress.update(task, advance=1, description=f"[white] Processing {subject}...")


def dcm2niix(input_path: str) -> str:
    """
    Converts DICOM images into Nifti images using dcm2niix
    :param input_path: Path to the folder with the dicom files to convert
    :return: str, Path to the folder with the converted nifti files
    """
    output_dir = os.path.dirname(input_path)

    # redirect standard output and standard error to discard output
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dicom2nifti.convert_directory(input_path, output_dir, compression=False, reorient=True)

    return output_dir


def remove_accents(unicode_filename):
    try:
        unicode_filename = str(unicode_filename).replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')
        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)
        return cleaned_filename
    except:
        return unicode_filename


def is_dicom_file(filename):
    try:
        pydicom.dcmread(filename)
        return True
    except pydicom.errors.InvalidDicomError:
        return False


def create_dicom_lookup(dicom_dir):
    """Create a lookup dictionary from DICOM files.

    Parameters:
    dicom_dir (str): The directory where DICOM files are stored.

    Returns:
    dict: A dictionary where the key is the anticipated filename that dicom2nifti will produce and
          the value is the modality of the DICOM series.
    """

    # a dictionary to store information from the DICOM files
    dicom_info = {}

    # loop over the DICOM files
    for filename in os.listdir(dicom_dir):
        full_path = os.path.join(dicom_dir, filename)
        if is_dicom_file(full_path):
            # read the DICOM file
            ds = pydicom.dcmread(full_path)

            # extract the necessary information
            series_number = ds.SeriesNumber if 'SeriesNumber' in ds else None
            series_description = ds.SeriesDescription if 'SeriesDescription' in ds else None
            sequence_name = ds.SequenceName if 'SequenceName' in ds else None
            protocol_name = ds.ProtocolName if 'ProtocolName' in ds else None
            series_instance_UID = ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else None
            if ds.Modality == 'PT':
                modality = 'PET'
            else:
                modality = ds.Modality

            # anticipate the filename dicom2nifti will produce and store the modality tag with it
            if series_number is not None:
                base_filename = remove_accents(series_number)
                if series_description is not None:
                    anticipated_filename = f"{base_filename}_{remove_accents(series_description)}.nii"
                elif sequence_name is not None:
                    anticipated_filename = f"{base_filename}_{remove_accents(sequence_name)}.nii"
                elif protocol_name is not None:
                    anticipated_filename = f"{base_filename}_{remove_accents(protocol_name)}.nii"
            else:
                anticipated_filename = f"{remove_accents(series_instance_UID)}.nii"

            dicom_info[anticipated_filename] = modality

    return dicom_info


def rename_nifti_files(nifti_dir, dicom_info):
    """Rename NIfTI files based on a lookup dictionary.

    Parameters:
    nifti_dir (str): The directory where NIfTI files are stored.
    dicom_info (dict): A dictionary where the key is the anticipated filename that dicom2nifti will produce and
                       the value is the modality of the DICOM series.
    """

    # loop over the NIfTI files
    for filename in os.listdir(nifti_dir):
        if filename.endswith('.nii'):
            # get the corresponding DICOM information
            modality = dicom_info.get(filename, '')
            if modality:  # only if the modality is found in the dicom_info dict
                # create the new filename
                new_filename = f"{modality}_{filename}"

                # rename the file
                os.rename(os.path.join(nifti_dir, filename), os.path.join(nifti_dir, new_filename))

                # delete the old name from the dictionary
                del dicom_info[filename]
