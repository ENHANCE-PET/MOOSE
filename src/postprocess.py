#!/usr/bin/env python
# -*- coding: utf-8 -*-

# **********************************************************************************************************************
# File: postprocess.py
# Project: MOOSE Version 1.0
# Created: 23.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: Lalith.Shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This module performs the postprocessing of the segmentation results.
# License: Apache 2.0
# **********************************************************************************************************************

import logging
import pathlib

import SimpleITK as sitk

import fileOp as fop
import greedy
import imageOp


def ct_segmentation(label_dir: str) -> None:
    """
    This function performs the postprocessing of the CT segmentation results.
    :param label_dir: The directory containing the CT segmentation results.
    :return:
    """
    logging.info(f"Post processing of the segmented CT image...")
    logging.info(f"- Removing overlays of organs from fat-muscle segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Organs*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Fat-Muscle*')[0],
                            out_image=fop.get_files(label_dir, 'Fat-Muscle*')[0])

    logging.info(f"- Removing overlays of Bones from fat-muscle segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Bones*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Fat-Muscle*')[0],
                            out_image=fop.get_files(label_dir, 'Fat-Muscle*')[0])
    logging.info(f"- Removing overlays of Psoas from fat-muscle segmentation")

    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Psoas*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Fat-Muscle*')[0],
                            out_image=fop.get_files(label_dir, 'Fat-Muscle*')[0])
    logging.info(f"Post processing of segmented CT image done")


def align_pet_ct(pet_img: str, ct_img: str, multilabel_seg: str) -> str:
    """
    This function aligns the PET and CT images.
    :param pet_img: The PET image.
    :param ct_img: The CT image.
    :param multilabel_seg: The multilabel segmentation from CT, that needs to be in alignment with PET.
    :return:
    """
    logging.info(f"Aligning PET and CT images...")
    aligned_multilabel_seg = pathlib.Path(multilabel_seg).stem.split(".")[0] + '_pet_aligned.nii.gz'
    greedy.deformable(fixed_img=pet_img, moving_img=ct_img, cost_function='NMI',
                      multi_resolution_iterations='100x50x25')
    greedy.resample(fixed_img=pet_img, moving_img=ct_img, resampled_moving_img=fop.add_prefix_rename(ct_img,
                                                                                                     'resampled_'),
                    registration_type='deformable', segmentation=multilabel_seg, resampled_seg=aligned_multilabel_seg)
    logging.info(f"Alignment of PET and CT images done and saved as {aligned_multilabel_seg}")
    return aligned_multilabel_seg


def brain_exists(ct_label: str) -> bool:
    """
    This function checks if the brain exists in the CT image.
    :param ct_label: Path of the ct multilabel image.
    :return: True if the brain exists in the CT image.
    """
    ct_mask = sitk.ReadImage(ct_label, sitk.sitkInt32)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(ct_mask, ct_mask)
    labels = []
    for label in stats.GetLabels():
        labels.append(label)
    return 4 in labels
