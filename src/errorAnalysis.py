#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ***********************************************************************************************************************
# File: errorAnalysis.py
# Project: MOOSE Version 1.0
# Created: 28.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: lalith.shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This module contains functions to perform error analysis on the segmentation results based on similarity
# space
# License: Apache 2.0
# **********************************************************************************************************************

import os
import pathlib

import SimpleITK as sitk
import mpire as mp
from mpire import WorkerPool
import logging
import constants as c
import fileOp as fop
import imageOp as iop
import pandas as pd

cpu_count = mp.cpu_count()


def split_multilabel_and_assign_names(multi_label_img: str, output_dir: str) -> None:
    """
    This function splits the multi-label image into individual label images
    :param multi_label_img: path to the multi-label image
    :param output_dir: path to the output directory
    :return: None
    """
    logging.info(f'Splitting the multi-label image: {multi_label_img} into individual label images')
    print(f'Splitting the multi-label image: {multi_label_img} into individual label images')
    logging.info(f'Split label images will be assigned to their label names and stored in: {output_dir}')
    print(f'Split label images will be assigned to their label names and stored in: {output_dir}')
    logging.info(f'Executing the splitting using multiple threads: {cpu_count}')
    print(f'Executing the splitting using multiple threads: {cpu_count}')
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(sitk.ReadImage(multi_label_img, sitk.sitkInt32))
    labels_present = list(shape_stats.GetLabels())
    with WorkerPool(n_jobs=cpu_count, shared_objects=(multi_label_img, output_dir),
                    start_method='fork', ) as pool:
        pool.map(split_multilabel_and_assign_names_func, labels_present, progress_bar=True)


def split_multilabel_and_assign_names_func(multi_label_param, label) -> None:
    """
    This function assigns names to the labels
    :param multi_label_param: (multi_label_img, output_dir)
    :param label: label that has to be assigned with a name based on c.ORGAN_INDEX
    :return: None (all the labels are assigned with names, in return all the labels will be binary images)
    """
    multi_label_img, output_dir = multi_label_param
    for key, value in c.ORGAN_INDEX.items():
        if key == label:
            organ_file = value + '.nii.gz'
            output_file = os.path.join(output_dir, organ_file)
            logging.info(f"{organ_file} will be scaled with the label: {label}")
            print(f"{organ_file} will be scaled with the label: {label}")
            iop.retain_labels(multi_label_img, [label], output_file)
            iop.binarize(output_file, output_file)
        else:
            continue
    return None


def split_dual_organs(dual_organ_dir: str, output_dir: str) -> None:
    """
    This function splits the mask image physically into left and right images.
    :param dual_organ_dir: Path to the directory that contains the individual dual organs as a single mask image (
    e.g. Left and right lung stored as Lung.nii.gz)
    :param output_dir: path to the output directory
    :return: None
    """
    logging.info(f'Physically splitting dual organs images from {dual_organ_dir} into left and right parts')
    logging.info(f"The output will be stored in {output_dir} and the original image will be deleted")
    logging.info(f'Executing the physical splitting using multiple threads: {cpu_count}')
    single_organ_paths = []
    for dual_organ in c.DUAL_ORGANS:
        if fop.get_files(dual_organ_dir, dual_organ + '*.nii.gz'):
            single_organ_paths.append(fop.get_files(dual_organ_dir, dual_organ + '*.nii.gz')[0])
        else:
            continue
    with WorkerPool(n_jobs=cpu_count, shared_objects=output_dir, start_method='fork', ) as pool:
        pool.map(split_dual_organs_func, single_organ_paths, progress_bar=True)


def split_dual_organs_func(output_dir: str, single_organ_mask: str) -> None:
    """
    This function splits the mask image physically into left and right images.
    :param output_dir: Path to the directory to store the split individual organs
    :param single_organ_mask: Path to the single organ mask file that needs to be split.
    :return: None
    """
    print(f'Physically splitting the mask image: {single_organ_mask} into left and right parts')
    iop.split_mask_to_left_right(binary_mask_path=single_organ_mask, out_dir=output_dir)
    print(f"Deleting the original mask image: {single_organ_mask}")
    fop.delete_files(str(pathlib.Path(single_organ_mask).parent), pathlib.Path(single_organ_mask).stem + '*')


def assign_labels_after_split(split_dir: str) -> None:
    """
    This function assigns labels to the split masks
    :param split_dir: Path to the split masks
    :return: None
    """
    logging.info(f'Assigning unique intensity labels to the split masks stored in {split_dir} based on predefined '
                 f'constants: {c.ORGAN_INDEX_SPLIT}')
    mask_files = fop.get_files(split_dir, '*.nii.gz')
    with WorkerPool(n_jobs=cpu_count) as pool:
        pool.map(assign_labels_after_split_func, mask_files, progress_bar=True)


def assign_labels_after_split_func(mask_file: str) -> None:
    """
    This function assigns labels to the split masks
    :param mask_file: Path to the mask file that needs to be assigned with a label
    :return: None
    """

    mask_name = pathlib.Path(mask_file).stem.split('.')[0]
    for key, value in c.ORGAN_INDEX_SPLIT.items():
        if value in mask_name:
            print(f"scaling {mask_name} with label: {key}")
            iop.scale_mask(mask_path=mask_file, out_path=mask_file, scale_factor=key)
        else:
            continue


def similarity_space(multi_label_img: str, out_dir: str, csv_out: str) -> None:
    """
    Function that performs the error analysis in similarity space
    :param multi_label_img: Path to the multi-label image
    :param out_dir: Path to the output directory
    :param csv_out: Path to the csv output file
    :return: None
    """
    sim_space_dir = fop.make_dir(out_dir, 'similarity-space')
    logging.info(f"Performing error analysis in similarity space for {multi_label_img}")
    split_multilabel_and_assign_names(multi_label_img, sim_space_dir)
    split_dual_organs(sim_space_dir, sim_space_dir)
    assign_labels_after_split(sim_space_dir)
    logging.info(f"Summing the split masks to get the final mask and store it in {sim_space_dir}")
    split_atlas = os.path.join(sim_space_dir, 'MOOSE-Split-unified-PET-CT-atlas.nii.gz')
    iop.sum_image_stack(sim_space_dir, '*nii.gz', split_atlas)
    logging.info(f"Measuring shape parameters for {split_atlas}")
    shape_parameters = iop.get_shape_parameters(split_atlas)
    normative_shape_parameters = pd.read_excel(c.NORMDB_DIR, engine="openpyxl")
    risk_score_df = pd.DataFrame(columns=['Labels', "Tissues", "Z-score", "Risk-of-segmentation-error"])
    overall_labels = normative_shape_parameters["Labels"].values.tolist()
    available_labels = shape_parameters.index.values.tolist()
    print(f"Available labels: {available_labels}")
    print(f"Overall labels: {overall_labels}")
    existing_labels = []
    existing_organs = []
    z_score = []
    risk = []
    for label in available_labels:
        if label in overall_labels:
            print(f"{label} is in the overall labels")
            existing_labels.append(label)
            existing_organs.append(normative_shape_parameters["Tissues"][normative_shape_parameters["Labels"] ==
                                                                         label].tolist()[0])
            z_deviation = (shape_parameters["Elongation"][label] - normative_shape_parameters["Mean"][
                normative_shape_parameters["Labels"] == label]).tolist()[0] / normative_shape_parameters["STD"][
                normative_shape_parameters["Labels"] == label].tolist()[0]
            z_score.append(z_deviation)
            if -1.5 <= z_deviation <= 1.5:
                risk.append('Low')
            else:
                risk.append('High')
        else:
            continue

    risk_score_df["Labels"] = existing_labels
    risk_score_df["Tissues"] = existing_organs
    risk_score_df["Z-score"] = z_score
    risk_score_df["Risk-of-segmentation-error"] = risk
    risk_score_df.to_csv(csv_out, index=False)
