#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module contains the constants that are used in the moosez.
#
# Usage:
# The variables in this module can be imported and used in other modules within the moosez.
#
# ----------------------------------------------------------------------------------------------------------------------

import os

from moosez import file_utilities

project_root = file_utilities.get_virtual_env_root()

NNUNET_RESULTS_FOLDER = os.path.join(project_root, 'models', 'nnunet_trained_models')
MOOSEZ_MODEL_FOLDER = os.path.join(NNUNET_RESULTS_FOLDER, 'nnUNet', '3d_fullres')
ALLOWED_MODALITIES = ['CT', 'FDG_PT', 'MR']
TEMP_FOLDER = 'temp'

# COLOR CODES
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RESET = '\033[0m'

# SUPPORTED TRACERS (limited patch)

TRACER_FDG = 'FDG'

# FOLDER NAMES

SEGMENTATIONS_FOLDER = 'segmentations'

# PREPROCESSING PARAMETERS

CLINICAL_VOXEL_SPACING = [1.5, 1.5, 1.5]
PRECLINICAL_VOXEL_SPACING = [0.15, 0.15, 0.15]
MATRIX_THRESHOLD = 200 * 200 * 600
Z_AXIS_THRESHOLD = 200
MARGIN_PADDING = 20
INTERPOLATION = 'bspline'
CHUNK_THRESHOLD = 200

# FILE NAMES PREFIX

ORGANS_MULTILABEL_SUFFIX = 'Organs-Multilabel-'
LUNGS_MULTILABEL_SUFFIX = 'Lungs-Multilabel-'
MULTILABEL_SUFFIX = 'MULTILABEL-'

# FILE NAMES

RESAMPLED_IMAGE_FILE_NAME = 'resampled_image.nii.gz'
RESAMPLED_MASK_FILE_NAME = 'resampled_mask.nii.gz'
