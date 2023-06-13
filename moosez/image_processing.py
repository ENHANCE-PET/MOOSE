#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
#         Sebastian Gutschmayer
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 05.06.2023
# Version: 2.0.0
#
# Description:
# This module handles image processing for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez for image processing.
#
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import nibabel
import SimpleITK


def resample(input_image_path: str, output_image_path: str, interpolation: str, desired_spacing: list) -> str:
    """
    Resamples an image to a new spacing.
    :param input_image_path: Path to the input image.
    :param output_image_path: Path to the output image.
    :param interpolation: The interpolation method to use.
    :param desired_spacing: The new spacing to use.
    :return: The path to the output image.
    """

    # Set interpolator for SimpleITK

    if interpolation == 'nearest':
        interpolation_method = SimpleITK.sitkNearestNeighbor
    elif interpolation == 'linear':
        interpolation_method = SimpleITK.sitkLinear
    elif interpolation == 'bspline':
        interpolation_method = SimpleITK.sitkBSpline
    else:
        raise ValueError('The interpolation method is not supported.')

    # Load the image and get necessary information
    input_image = nibabel.load(input_image_path)
    image_data = input_image.get_fdata()
    image_header = input_image.header
    image_affine = input_image.affine
    original_spacing = image_header.get_zooms()
    original_size = image_data.shape
    translation_vector = image_affine[:3, 3]
    rotation_matrix = image_affine[:3, :3]

    # Convert to SimpleITK image format
    image_data_swapped_axes = image_data.swapaxes(0, 2)
    sitk_input_image = SimpleITK.GetImageFromArray(image_data_swapped_axes)
    sitk_input_image.SetSpacing([spacing.item() for spacing in original_spacing])
    axis_flip_matrix = np.diag([-1, -1, 1])
    sitk_input_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
    sitk_input_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).ravel())

    # Resample the image to the desired spacing
    desired_spacing = np.array(desired_spacing)
    new_size = [round(original_size[i] * (original_spacing[i] / desired_spacing[i])) for i in range(len(original_size))]
    resampled_sitk_image = SimpleITK.Resample(sitk_input_image, new_size, SimpleITK.Transform(), interpolation_method,
                                              sitk_input_image.GetOrigin(), desired_spacing.tolist(),
                                              sitk_input_image.GetDirection(), 0.0, sitk_input_image.GetPixelIDValue())

    # Save the resampled image to disk
    SimpleITK.WriteImage(resampled_sitk_image, output_image_path)

    return output_image_path


def get_spacing(image_path: str) -> list:
    """
    Returns the spacing of an image.
    :param image_path: Path to the image.
    :return: The spacing of the image.
    """
    image = nibabel.load(image_path)
    image_header = image.header
    spacing = image_header.get_zooms()
    return [spacing.item() for spacing in spacing]
