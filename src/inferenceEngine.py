#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ***********************************************************************************************************************
# File: inferenceEngine.py
# Project: MOOSE Version 0.1.0
# Created: 23.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: Lalith.Shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This module performs the inference using nnUNet, basically just a sugar layer.
# License: Apache 2.0
# **********************************************************************************************************************
import logging
import os
import pathlib
import re
import subprocess
import constants as c
import fileOp as fop
import imageOp
import postProcessing
from halo import Halo


def model_number(tissue_type: str) -> int:
    """
    Returns the model number for the given tissue type
    :param tissue_type: 'Organs' or 'Brain' or 'Psoas' or 'Fat-Muscle' or 'Bones
    :return: model number for the given tissue type which is used in nnUNet
    """
    if tissue_type == 'Organs':
        return 123
    elif tissue_type == 'Psoas':
        return 126
    elif tissue_type == 'Brain':
        return 327
    elif tissue_type == 'Fat-Muscle':
        return 427
    elif tissue_type == 'Bones':
        return 517
    else:
        return 0


def segment_tissue(nifti_img: str, out_dir: str, tissue_type: str) -> None:
    """
    Segment a given nifti image using nnUNet
    :param nifti_img: path to the nifti image
    :param out_dir: path to the output directory
    :param tissue_type: tissue type to segment
    :return: None
    """
    model = str(model_number(tissue_type))
    cmd_to_run = f"nnUNet_predict -i {re.escape(str(pathlib.Path(nifti_img).parents[0]))} -o {re.escape(out_dir)} -t " \
                 f"{model} -m 3d_fullres --fold all "
    subprocess.run(cmd_to_run, shell=True, capture_output=True)


def segment_ct(nifti_img: str, out_dir: str) -> str:
    """
    Segment a given nifti image using nnUNet
    :param nifti_img: path to the nifti CT image
    :param out_dir: path to the output directory
    :return: None
    """
    nifti_img_ext = fop.add_suffix_rename(nifti_img, '0000')
    ct_file = fop.compress_file(nifti_img_ext)
    logging.info(f"CT image to be segmented: {re.escape(ct_file)}")
    print(f'- CT image to be segmented: {re.escape(ct_file)}')
    logging.info(f"Output directory: {re.escape(out_dir)}")

    logging.info(f"Segmenting abdominal organs...")
    spinner = Halo(text=f"Segmenting abdominal organs from {ct_file}", spinner='dots')
    spinner.start()
    segment_tissue(ct_file, out_dir, 'Organs')
    spinner.succeed(text=f"Segmented abdominal organs from {ct_file}")
    print('OUT LABEL:', fop.get_files(out_dir, pathlib.Path(nifti_img).stem + '*'))
    out_label = fop.get_files(out_dir, pathlib.Path(nifti_img).stem + '*')[0]
    fop.add_prefix_rename(out_label, 'Organs')
    logging.info(f"Initiating post processing of abdominal organs...")
    imageOp.replace_intensity(image_to_replace=fop.get_files(out_dir, 'Organs*')[0], intensity=[12, 0, 13, 12],
                              out_image=fop.get_files(out_dir, 'Organs*')[0])
    logging.info(f"- Removing skeleton...replacing skeleton intensity (12) with 0")
    logging.info(f"- Replacing lung intensity (13) with 12")
    logging.info(f"Abdominal organs segmented and saved in {fop.get_files(out_dir, 'Organs*')[0]}")

    logging.info(f"Segmenting Bones...")
    spinner = Halo(text=f"Segmenting Bones from {ct_file}", spinner='dots')
    spinner.start()
    segment_tissue(ct_file, out_dir, 'Bones')
    spinner.succeed(text=f"Segmented Bones from {ct_file}")
    fop.add_prefix_rename(out_label, 'Bones')
    imageOp.shift_intensity(image_to_shift=fop.get_files(out_dir, 'Bones*')[0],
                            shift_amount=c.NUM_OF_ORGANS,
                            out_image=fop.get_files(out_dir, 'Bones*')[0])
    imageOp.replace_intensity(image_to_replace=fop.get_files(out_dir, 'Bones*')[0], intensity=[c.NUM_OF_ORGANS, 0],
                              out_image=fop.get_files(out_dir, 'Bones*')[0])
    logging.info(f"Bones segmented and saved in {fop.get_files(out_dir, 'Bones*')[0]}")

    logging.info(f"Segmenting skeletal muscle, subcutaneous and visceral fat...")
    spinner = Halo(text=f"Segmenting skeletal muscle, subcutaneous and visceral fat from {ct_file}", spinner='dots')
    spinner.start()
    segment_tissue(ct_file, out_dir, 'Fat-Muscle')
    spinner.succeed(text=f"Segmented skeletal muscle, subcutaneous and visceral fat from {ct_file}")
    fop.add_prefix_rename(out_label, 'Fat-Muscle')
    imageOp.shift_intensity(image_to_shift=fop.get_files(out_dir, 'Fat-Muscle*')[0],
                            shift_amount=c.NUM_OF_ORGANS + c.NUM_OF_BONES,
                            out_image=fop.get_files(out_dir, 'Fat-Muscle*')[0])
    imageOp.replace_intensity(image_to_replace=fop.get_files(out_dir, 'Fat-Muscle*')[0], intensity=[c.NUM_OF_ORGANS +
                                                                                                    c.NUM_OF_BONES, 0],
                              out_image=fop.get_files(out_dir, 'Fat-Muscle*')[0])
    logging.info(f"Fat-Muscle segmented and saved in {fop.get_files(out_dir, 'Fat-Muscle*')[0]}")

    logging.info(f"Segmenting psoas and assigning the right label intensity...")
    spinner = Halo(text=f"Segmenting psoas from {ct_file}", spinner='dots')
    spinner.start()
    segment_tissue(ct_file, out_dir, 'Psoas')
    spinner.succeed(text=f"Segmented psoas from {ct_file}")
    fop.add_prefix_rename(out_label, 'Psoas')
    imageOp.shift_intensity(image_to_shift=fop.get_files(out_dir, 'Psoas*')[0],
                            shift_amount=c.NUM_OF_ORGANS + c.NUM_OF_BONES + c.NUM_OF_FAT_MUSCLE,
                            out_image=fop.get_files(out_dir, 'Psoas*')[0])
    imageOp.replace_intensity(image_to_replace=fop.get_files(out_dir, 'Psoas*')[0], intensity=[c.NUM_OF_ORGANS +
                                                                                               c.NUM_OF_BONES +
                                                                                               c.NUM_OF_FAT_MUSCLE, 0],
                              out_image=fop.get_files(out_dir, 'Psoas*')[0])
    logging.info(f"Psoas segmented and saved in {fop.get_files(out_dir, 'Psoas*')[0]}")
    postProcessing.ct_segmentation(label_dir=out_dir)
    logging.info(f"Merging all non-cerebral tissues segmented from CT...")
    imageOp.sum_image_stack(out_dir, '*nii.gz', os.path.join(
        out_dir, 'MOOSE-Non-cerebral-tissues-CT-' + pathlib.Path(out_dir).parents[0].stem + '.nii.gz'))
    logging.info(f"Non-cerebral tissues segmented and saved in {fop.get_files(out_dir, 'MOOSE*CT*nii.gz')[0]}")
    return fop.get_files(out_dir, 'MOOSE*CT*nii.gz')[0]


def segment_pt(nifti_img: str, out_dir: str) -> str:
    """
    Segment a given nifti image using nnUNet
    :param nifti_img: path to the nifti PT image
    :param out_dir: path to the output directory
    :return: path where the segmented pt image is saved
    """
    pt_file = nifti_img
    logging.info(f"PT image to be segmented: {pt_file}")
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Segmenting brain...")
    spinner = Halo(text=f"Segmenting brain from {pt_file}", spinner='dots')
    spinner.start()
    segment_tissue(pt_file, out_dir, 'Brain')
    spinner.succeed(text=f"Segmented brain from {pt_file}")
    out_label = fop.get_files(out_dir, c.CROPPED_BRAIN_FROM_PET[:13] + '*')[0]
    fop.add_prefix_rename(out_label, 'Brain')
    imageOp.shift_intensity(image_to_shift=fop.get_files(out_dir, 'Brain*')[0],
                            shift_amount=c.NUM_OF_ORGANS + c.NUM_OF_BONES + c.NUM_OF_FAT_MUSCLE + c.NUM_OF_PSOAS,
                            out_image=fop.get_files(out_dir, 'Brain*')[0])
    imageOp.replace_intensity(image_to_replace=fop.get_files(out_dir, 'Brain*')[0], intensity=[
        c.NUM_OF_ORGANS + c.NUM_OF_BONES + c.NUM_OF_FAT_MUSCLE + c.NUM_OF_PSOAS, 0],
                              out_image=fop.get_files(out_dir, 'Brain*')[0])
    logging.info(f"Brain segmented and saved in {fop.get_files(out_dir, 'Brain*')[0]}")
    return fop.get_files(out_dir, 'Brain*')[0]
