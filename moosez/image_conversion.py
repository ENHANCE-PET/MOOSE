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
# This module handles image conversion for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform image conversion.
#
# ----------------------------------------------------------------------------------------------------------------------

import subprocess
import re
import logging
from halo import Halo


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
