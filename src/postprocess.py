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
import os
import pathlib

import SimpleITK as sitk

import fileOp as fop
import greedy
import imageOp


def pt_segmentation(label_dir: str) -> None:
    """
    This function performs processing of the PT segmentation results
    :param label_dir: Path of the directory that contains the PT segmentation results
    :return: None
    """
    logging.info("Performing postprocessing of the PT segmentation results")
    logging.info("Removing the overlays of PT brain segmentation results from the PT aligned non-cerebral CT "
                 "segmentations")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Brain*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'pet_aligned*')[0], out_image=
                            fop.get_files(label_dir, 'pet_aligned*')[0])


def ct_segmentation(label_dir: str) -> None:
    """
    This function performs the postprocessing of the CT segmentation results.
    :param label_dir: The directory containing the CT segmentation results.
    :return:
    """
    logging.info(f"Post processing of the segmented CT image...")
    logging.info(f"- Removing overlays of organs from Bone segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Organs*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Bones*')[0],
                            out_image=fop.get_files(label_dir, 'Bones*')[0])
    logging.info(f"- Removing overlays of organs from fat-muscle segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Organs*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Fat-Muscle*')[0],
                            out_image=fop.get_files(label_dir, 'Fat-Muscle*')[0])
    logging.info(f"- Removing overlays of organs from Psoas segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Organs*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Psoas*')[0],
                            out_image=fop.get_files(label_dir, 'Psoas*')[0])

    logging.info(f"- Removing overlays of Bones from fat-muscle segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Bones*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Fat-Muscle*')[0],
                            out_image=fop.get_files(label_dir, 'Fat-Muscle*')[0])
    logging.info(f"- Removing overlays of Bones from Psoas segmentation")
    imageOp.remove_overlays(reference_image=fop.get_files(label_dir, 'Bones*')[0],
                            image_to_remove_overlays=fop.get_files(label_dir, 'Psoas*')[0],
                            out_image=fop.get_files(label_dir, 'Psoas*')[0])
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
    aligned_multilabel_seg = fop.add_prefix(multilabel_seg, 'pet_aligned_')
    greedy.affine(fixed_img=pet_img, moving_img=ct_img, cost_function='NMI',
                  multi_resolution_iterations='100x50x25')
    greedy.resample(fixed_img=pet_img, moving_img=ct_img, resampled_moving_img=fop.add_prefix(ct_img,
                                                                                              'pet_aligned_'),
                    registration_type='affine', segmentation=multilabel_seg, resampled_seg=aligned_multilabel_seg)
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


def merge_pet_ct_segmentations(pet_seg: str, ct_seg: str, out_seg: str) -> str:
    """
    Merge the pet and ct segmentation to one unified segmentation file. The returned file will be in PT space
    :param pet_seg: Path of the PET segmentation file
    :param ct_seg: Path of the CT segmentation file
    :param out_seg: Path of the unified segmentation file
    :return: Path of the unified segmentation file
    """
    logging.info('Removing whole-brain segmentation from CT segmentation to merge the 83 subregions from PET')
    out_dir = str(pathlib.Path(out_seg).parents[0])
    imageOp.retain_labels(image_to_retain_labels=ct_seg, labels_to_retain=[4], out_image=os.path.join(out_dir,
                                                                                                      'whole_brain_ct_segmentation.nii.gz'))
    imageOp.replace_intensity(image_to_replace=ct_seg, intensity=[4, 0], out_image=ct_seg)
    logging.info("Reslicing (to identity) PET segmentation, segmentations will be in pet voxel space...")
    imageOp.reslice_identity(reference_image=ct_seg, image_to_reslice=pet_seg, out_resliced_image=pet_seg,
                             interpolation='NearestNeighbor')
    pt_segmentation(label_dir=out_dir)
    logging.info("Merging PET and CT segmentations...")
    imageOp.sum_images(img_list=[pet_seg, ct_seg], out_img=out_seg)
    logging.info("Merging PET and CT segmentations done")
    return out_seg
