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


import os
import SimpleITK
import dicom2nifti
import pydicom
import multiprocessing as mp
import concurrent.futures
import subprocess
from typing import Union
from moosez import system


def dicom_to_nifti(input_directory: str, output_directory: Union[str, None] = None, compression: bool = False) -> Union[str, None]:
    if output_directory is None:
        output_directory = input_directory

    input_directory_files = [file for file in os.listdir(input_directory) if
                             is_dicom_file(os.path.join(input_directory, file))]
    if len(input_directory_files) < 1:
        return None

    if compression:
        extension = ".nii.gz"
        _compress = "y"
    else:
        extension = ".nii"
        _compress = "n"

    ds = pydicom.dcmread(os.path.join(input_directory, input_directory_files[0]), stop_before_pixels=True)
    modality = ds.Modality
    output_file_name = f"{modality}_{os.path.basename(input_directory)}"
    output_file_path = os.path.join(output_directory, f"{output_file_name}{extension}")
    if os.path.exists(output_file_path):
        return None

    try:
        dicom2nifti.dicom_series_to_nifti(input_directory, output_file_path, reorient_nifti=True)
        return output_file_path
    except Exception:
        try:
            dcm2niix_cmd = ["dcm2niix", "-o", output_directory, "-f", output_file_name, "-z", _compress, "-b", "n", "-v", "0", input_directory]
            subprocess.run(dcm2niix_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_file_path

        except Exception:
            return None


def non_nifti_to_nifti(input_path: str, output_directory: Union[str, None] = None) -> None:
    """
    Converts any image format known to ITK to NIFTI

        :param input_path: The path to the directory or filename to convert to nii.gz.
        :type input_path: str

        :param output_directory: Optional. The output directory to write the image to. If not specified, the
                                 output image will be written to the same directory as the input image.
        :type output_directory: str

        :return: None
        :rtype: None

        :raises: FileNotFoundError if the input path does not exist.

        Usage:
        This function can be used to convert any image format known to ITK to NIFTI. If the input path is a directory,
        the function will convert all images in the directory to NIFTI format. If the input path is a file,
        the function will convert the file to NIFTI format. The output image will be written to the specified
        output directory or to the same directory as the input image if no output directory is specified.
    """

    if not os.path.exists(input_path):
        return

    if output_directory is None:
        output_directory = os.path.dirname(input_path)

    # Processing a directory
    if os.path.isdir(input_path):
        dicom_to_nifti(input_path, output_directory)
        return

    # Processing a file
    if os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        if filename.startswith('.') or filename.endswith(('.nii.gz', '.nii')):
            return

        output_image = SimpleITK.ReadImage(input_path)
        output_image_basename = f"{os.path.splitext(filename)[0]}.nii"
        output_image_path = os.path.join(output_directory, output_image_basename)
        SimpleITK.WriteImage(output_image, output_image_path)


def standardize_subject(parent_dir: str, subject: str):
    subject_path = os.path.join(parent_dir, subject)
    if not os.path.isdir(subject_path):
        return

    images = os.listdir(subject_path)
    for image in images:
        image_path = os.path.join(subject_path, image)
        path_is_valid = os.path.isdir(image_path) or os.path.isfile(image_path)
        path_is_valid = path_is_valid and ("moosez" not in os.path.basename(image_path))
        if path_is_valid:
            non_nifti_to_nifti(image_path)


def standardize_to_nifti(parent_dir: str, output_manager: system.OutputManager) -> None:
    """
    Converts all non-NIfTI images in a parent directory and its subdirectories to NIfTI format.

    :param parent_dir: The path to the parent directory containing the images to convert.
    :type parent_dir: str
    :param output_manager: The output manager to handle console and log output.
    :type output_manager: system.OutputManager
    :return: None
    """
    # Get a list of all subdirectories in the parent directory
    subjects = os.listdir(parent_dir)
    subjects = [subject for subject in subjects if os.path.isdir(os.path.join(parent_dir, subject))]

    # Convert all non-NIfTI images in each subdirectory to NIfTI format
    progress = output_manager.create_progress_bar()
    with progress:
        task = progress.add_task("[white] Processing subjects...", total=len(subjects))

        mp_context = mp.get_context('spawn')
        max_workers = mp.cpu_count() // 4 if mp.cpu_count() > 4 else 1
        max_workers = max_workers if max_workers <= 32 else 32
        output_manager.log_update(f"Number of workers: {max_workers}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            futures = {executor.submit(standardize_subject, parent_dir, subject): subject for subject in subjects}
            for future in concurrent.futures.as_completed(futures):
                completed_subject = futures[future]
                progress.update(task, advance=1, description=f"[white] Processing {completed_subject}...")


def is_dicom_file(filename: str) -> bool:
    """
    Checks if a file is a DICOM file.

    :param filename: The path to the file to check.
    :type filename: str
    :return: True if the file is a DICOM file, False otherwise.
    :rtype: bool
    """
    try:
        pydicom.dcmread(filename, stop_before_pixels=True)
        return True
    except pydicom.errors.InvalidDicomError:
        return False