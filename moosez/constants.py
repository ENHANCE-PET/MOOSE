#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the constants that are used in the moosez.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 3.0.0
"""

ALLOWED_MODALITY_CONFIGURATIONS = {'CT': [None],
                                   'PT': ['FDG', 'PSMA'],
                                   'MR': [None, 'T1', 'T2'],
                                   'NM': ['LU'],
                                   'ST': ['LU']}

# COLOR CODES
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RED = '\033[38;5;196m'
ANSI_RESET = '\033[0m'

# FOLDER NAMES
SEGMENTATIONS_FOLDER = 'segmentations'
STATS_FOLDER = 'stats'
SINGLE_LABELS_FOLDER = 'individual_segmentations'

# MODELS
KEY_FOLDER_NAME = "folder_name"
KEY_URL = "url"
# optional per-model settings, see models.MODEL_METADATA
KEY_READER = "reader"
KEY_ORIENTATION = "orientation"
KEY_INPUT_RESAMPLING = "input_resampling"
KEY_ROI = "roi"
INPUT_RESAMPLING_CUBIC = "cubic"
# moosez reads every image with SimpleITK, so a model runs with one of these nnU-Net readers. A model whose plans name
# another reader that gives the same arrays runs with its SimpleITK twin: NibabelIOWithReorient and
# SimpleITKIOWithReorient both give the image in RAS as a (z, y, x) array with its spacing (bitwise equal on ENHANCE.PET
# CTs; headers whose qform and sform disagree, or that ITK cannot read as orthonormal, are not covered).
SUPPORTED_READERS = ("SimpleITKIO", "SimpleITKIOWithReorient")
SIMPLEITK_READER_FOR = {"NibabelIOWithReorient": "SimpleITKIOWithReorient"}
FILE_NAME_DATASET_JSON = "dataset.json"
FILE_NAME_PLANS_JSON = "plans.json"

# Weights that are placed into the models directory by hand instead of being downloaded.
LOCAL_URL_PREFIX = "local://"
# Runs when no model is named.
DEFAULT_MODEL = "clin_ct_anatomy"


ENHANCE_URLS = {
    "CT_details.xlsx": "https://download.scidb.cn/download?fileId=66dc051cacd26e61ffb057a26eb0fc22&path=/V2/ENHANCE-PET_1.6k/CT-details.xlsx&fileName=CT-details.xlsx",
    "PT_details.xlsx": "https://download.scidb.cn/download?fileId=baa3944cedd7483ec7ca3a131b93336e&path=/V2/ENHANCE-PET_1.6k/PT-details.xlsx&fileName=PT-details.xlsx",
    "labels.json": "https://download.scidb.cn/download?fileId=f22b5214da66fde206716ce4cb6beec9&path=/V2/ENHANCE-PET_1.6k/labels.json&fileName=labels.json",
    "body_composition_labels.zip": "https://download.scidb.cn/download?fileId=f37bc4ada0c4b727e54712c7903a5000&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Body-Composition.zip&fileName=Body-Composition.zip",
    "cardiac_labels.zip": "https://download.scidb.cn/download?fileId=a3ab7f1fa36f1916a7257229f360c53c&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Cardiac.zip&fileName=Cardiac.zip",
    "muscle_labels.zip": "https://download.scidb.cn/download?fileId=3349b8894c918b484350e159a4d9aa87&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Muscles.zip&fileName=Muscles.zip",
    "organs_labels.zip": "https://download.scidb.cn/download?fileId=01370960ad902c33cb77dd86f9a6cd30&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Organs.zip&fileName=Organs.zip",
    "peripheral_bones_labels.zip": "https://download.scidb.cn/download?fileId=8bbfada7664c40646c1fdd5b79acc76d&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Peripheral-Bones.zip&fileName=Peripheral-Bones.zip",
    "ribs_labels.zip": "https://download.scidb.cn/download?fileId=7742aa18a9bdc9bcfcc5bd58935c11f3&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Ribs.zip&fileName=Ribs.zip",
    "vertebrae_labels.zip": "https://download.scidb.cn/download?fileId=6094a2cc1db705c1fbd93f6ed93f3225&path=/V2/ENHANCE-PET_1.6k/imaging-data/ground-truth/Vertebrae.zip&fileName=Vertebrae.zip",
    "PT_images.zip": "https://download.scidb.cn/download?fileId=5050349884db27c7962d3eab3f97311a&path=/V2/ENHANCE-PET_1.6k/imaging-data/images/PT_0001-1683.zip&fileName=PT_0001-1683.zip",
    "CT_images_0001-1000.zip": "https://download.scidb.cn/download?fileId=3d15b56ca4bce7de7ff5cd5dfbad8c11&path=/V2/ENHANCE-PET_1.6k/imaging-data/images/CT_0001-1000.zip&fileName=CT_0001-1000.zip",
    "CT_images_1000-1683.zip": "https://download.scidb.cn/download?fileId=02c276fd440cc5f104a6b655cc3ac6f0&path=/V2/ENHANCE-PET_1.6k/imaging-data/images/CT_1001-1683.zip&fileName=CT_1001-1683.zip"
}


USAGE_MESSAGE = """
Usage:
  moosez -i <IMAGE_FILES_OR_FOLDERS> -o <OUTPUT_DIRECTORY> [-m <MODEL_NAMES>]
  moosez -d <MAIN_DIRECTORY> -m <MODEL_NAMES> -b
Example:
  moosez -i /Documents/CTs/ -o /Documents/CT_segmentations/
  moosez -d /Documents/Data_to_moose/
  moosez -d /Documents/Data_to_moose/ -m clin_ct_organs
  moosez -d /Documents/Data_to_moose/ -m clin_ct_organs clin_ct_ribs clin_ct_muscles 
  moosez -d /Documents/Data_to_moose/ -m clin_ct_organs clin_ct_ribs clin_ct_muscles -h 2

Description:
  MOOSE (Multi-organ objective segmentation) - A data-centric AI solution that
  generates multilabel organ segmentations for systemic TB whole-person research."""
