#!/usr/bin/env python
# -*- coding: utf-8 -*-


# ***********************************************************************************************************************
# File: greedy.py
# Project: MOOSE Version 1.0
# Created: 21.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: Lalith.Shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This contains the registration routines for aligning PET and CT images (part of falcon 🦅).
# License: Apache 2.0
# **********************************************************************************************************************

# Libraries to import

import logging
import os
import pathlib
import re
import subprocess
import sys


def rigid(fixed_img: str, moving_img: str, cost_function: str, multi_resolution_iterations: str) -> None:
    """ Performs rigid registration between a fixed and moving image using the greedy registration toolkit.
    :param fixed_img: Reference image
    :param moving_img: Moving image
    :param cost_function: Cost function
    :param multi_resolution_iterations: Amount of iterations for each resolution level
    :return none
    """
    logging.info(" ")
    cmd_to_run = f"greedy -d 3 -a -i " \
                 f"{re.escape(fixed_img)} {re.escape(moving_img)} -ia-image-centers -dof 6 -o rigid.mat -n " \
                 f"{multi_resolution_iterations} " \
                 f"-m {cost_function}"
    logging.info(f"Registration type: Rigid")
    logging.info(f"Reference image: {re.escape(fixed_img)}")
    logging.info(f"Moving image: {re.escape(moving_img)}")
    logging.info(f"Cost function: {cost_function}")
    logging.info(f"Initial alignment: Image centers")
    logging.info(f"Multi-resolution level iterations: {multi_resolution_iterations}")
    logging.info(f"Transform file generated: rigid.mat")
    logging.info(" ")
    subprocess.run(cmd_to_run, shell=True, capture_output=True)
    print("Rigid registration complete")


def affine(fixed_img: str, moving_img: str, cost_function: str, multi_resolution_iterations: str) -> None:
    """ Performs affine registration between a fixed and moving image using the greedy registration toolkit.
    :param fixed_img: Reference image
    :param moving_img: Moving image
    :param cost_function: Cost function
    :param multi_resolution_iterations: Amount of iterations for each resolution level

    :return none
    """
    logging.info(" ")
    cmd_to_run = f"greedy -d 3 -a -i {re.escape(fixed_img)} {re.escape(moving_img)} -ia-image-centers -dof 12 -o " \
                 f"affine.mat -n " \
                 f"{multi_resolution_iterations} " \
                 f"-m {cost_function} "
    logging.info(f"- Registration type: Affine")
    logging.info(f"- Reference image: {re.escape(fixed_img)}")
    logging.info(f"- Moving image: {re.escape(moving_img)}")
    logging.info(f"- Cost function: {cost_function}")
    logging.info(f"- Initial alignment: Image centers")
    logging.info(f"- Multi-resolution level iterations: {multi_resolution_iterations}")
    logging.info(f"- Transform file generated: affine.mat")
    logging.info(" ")
    subprocess.run(cmd_to_run, shell=True, capture_output=True)


def deformable(fixed_img: str, moving_img: str, cost_function: str, multi_resolution_iterations: str) -> None:
    """
    Performs deformable registration between a fixed and moving image using the greedy registration toolkit.
    :param fixed_img: Reference image
    :param moving_img: Moving image
    :param cost_function: Cost function
    :param multi_resolution_iterations: Amount of iterations for each resolution level
    :return:
    """
    logging.info(" ")
    logging.info("Performing affine registration for initial global alignment")
    affine(fixed_img, moving_img, cost_function, multi_resolution_iterations)
    cmd_to_run = f"greedy -d 3 -m {cost_function} -i {re.escape(fixed_img)} {re.escape(moving_img)} -it affine.mat -o " \
                 f"warp.nii.gz -oinv " \
                 f"inverse_warp.nii.gz -n {multi_resolution_iterations}"
    logging.info("Performing deformable registration for local alignment")
    logging.info(f"- Registration type: deformable")
    logging.info(f"- Reference image: {re.escape(fixed_img)}")
    logging.info(f"- Moving image: {re.escape(moving_img)}")
    logging.info(f"- Cost function: {cost_function}")
    logging.info(f"- Initial alignment: based on affine.mat")
    logging.info(f"- Multiresolution level iterations: {multi_resolution_iterations}")
    logging.info(f"- Deformation field generated: warp.nii.gz + inverse_warp.nii.gz")
    logging.info(' ')
    subprocess.run(cmd_to_run, shell=True, capture_output=True)
    print("Deformable registration complete")


def registration(fixed_img: str, moving_img: str, registration_type: str, multi_resolution_iterations: str) -> None:
    """
    Registers the fixed and the moving image using the greedy registration toolkit based on the user given cost function
    :param fixed_img: Reference image
    :param moving_img: Moving image
    :param registration_type: Type of registration ('rigid', 'affine' or 'deformable')
    :param multi_resolution_iterations: Amount of iterations for each resolution level
    :return: None
    """
    logging.info(" ")
    logging.info("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    logging.info(f"Aligning: {pathlib.Path(moving_img).name} -> {pathlib.Path(fixed_img).name}")
    logging.info(f"Registration mode: {registration_type}")
    logging.info("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    if registration_type == 'rigid':
        rigid(fixed_img, moving_img, cost_function='NMI', multi_resolution_iterations=multi_resolution_iterations)
    elif registration_type == 'affine':
        affine(fixed_img, moving_img, cost_function='NMI', multi_resolution_iterations=multi_resolution_iterations)
    elif registration_type == 'deformable':
        deformable(fixed_img, moving_img, cost_function='NCC 2x2x2',
                   multi_resolution_iterations=multi_resolution_iterations)
    else:
        sys.exit("Registration type not supported!")


def resample(fixed_img: str, moving_img: str, resampled_moving_img: str, registration_type: str, segmentation="",
             resampled_seg="") -> None:
    """
    Resamples a moving image to match the resolution of a fixed image.
    :param fixed_img: Reference image
    :param moving_img: Moving image
    :param resampled_moving_img: Resampled moving image
    :param registration_type: 'rigid', 'affine', or 'deformable'
    :param segmentation: Mask image corresponding to moving image that needs to be resampled to match reference image
    :param resampled_seg: Resampled mask image
    :return: None
    """
    if registration_type == 'rigid':
        if segmentation and resampled_seg:
            cmd_to_run = f"greedy -d 3 -rf {re.escape(fixed_img)} -ri NN -rm {re.escape(moving_img)} " \
                         f"{re.escape(resampled_moving_img)} -ri LABEL " \
                         f"0.2vox -rm {re.escape(segmentation)} {re.escape(resampled_seg)} -r rigid.mat"
        else:
            cmd_to_run = f"greedy -d 3 -rf {re.escape(fixed_img)} -ri NN -rm {re.escape(moving_img)} " \
                         f"{re.escape(resampled_moving_img)} -r rigid.mat "
    elif registration_type == 'affine':
        if segmentation and resampled_seg:
            cmd_to_run = f"greedy -d 3 -rf {re.escape(fixed_img)} -ri NN -rm {re.escape(moving_img)} " \
                         f"{re.escape(resampled_moving_img)} -ri LABEL " \
                         f"0.2vox -rm {re.escape(segmentation)} {re.escape(resampled_seg)} -r affine.mat"
        else:
            cmd_to_run = f"greedy -d 3 -rf {re.escape(fixed_img)} -ri NN -rm {re.escape(moving_img)} " \
                         f"{re.escape(resampled_moving_img)} -r affine.mat"
    elif registration_type == 'deformable':
        if segmentation and resampled_seg:
            cmd_to_run = f"greedy -d 3 -rf {re.escape(fixed_img)} -ri NN -rm {re.escape(moving_img)} " \
                         f"{re.escape(resampled_moving_img)} -ri LABEL " \
                         f"0.2vox -rm {re.escape(segmentation)} {re.escape(resampled_seg)} -r warp.nii.gz affine.mat"
        else:
            cmd_to_run = f"greedy -d 3 -rf {re.escape(fixed_img)} -ri NN -rm {re.escape(moving_img)} " \
                         f"{re.escape(resampled_moving_img)} -r warp.nii.gz " \
                         f"affine.mat"
    else:
        sys.exit("Registration type not supported!")
    subprocess.run(cmd_to_run, shell=True, capture_output=True)
    logging.info(f"Resampling parameters:")
    logging.info(f"- Reference image: {re.escape(fixed_img)}")
    logging.info(f"- Moving image: {re.escape(moving_img)}")
    logging.info(f"- Resampled moving image: {resampled_moving_img}")
    logging.info(f"- Segmentation: {segmentation}")
    logging.info(f"- Resampled segmentation: {resampled_seg}")
    logging.info(f"- Interpolation scheme for resampling: Nearest neighbor for images and segmentations")
    logging.info(' ')
