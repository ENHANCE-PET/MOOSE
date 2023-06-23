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

import SimpleITK
import nibabel
from nibabel import Nifti1Image
from moosez import constants
import numpy as np
import dask.array as da


def chunk_along_axis(axis: int) -> int:
    """

    Args:
        axis: the axis length as integer

    Returns:
        object: the determined integer values that splits the axis evenly
    """
    split = 2
    result = axis / split
    rest = axis % split

    while result >= constants.CHUNK_THRESHOLD or rest != 0:
        split += 1
        result = axis / split
        rest = axis % split

    return split


def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: tuple, interpolation_method: str,
                             output_spacing: tuple = (1.5, 1.5, 1.5)) -> da.array:
    """
    Resamples a dask array chunk
    :param interpolation_method: SimpleITK interpolation type
    :param image_chunk: The chunk (part of an image) to be resampled
    :param input_spacing: The original spacing of the chunk (part of an image)
    :param output_spacing: The desired output spacing
    :return: The resampled chunk (part of an image)
    """

    sitk_image_chunk = SimpleITK.GetImageFromArray(image_chunk)
    sitk_image_chunk.SetSpacing(input_spacing)
    input_size = sitk_image_chunk.GetSize()
    output_size = [round(input_size[i] * (input_spacing[i] / output_spacing[i])) for i in range(len(input_size))]

    if all(x == 0 for x in input_size):
        return image_chunk

    resampled_sitk_image = SimpleITK.Resample(sitk_image_chunk, output_size, SimpleITK.Transform(),
                                              interpolation_method,
                                              sitk_image_chunk.GetOrigin(), output_spacing,
                                              sitk_image_chunk.GetDirection(), 0.0, sitk_image_chunk.GetPixelIDValue())

    resampled_array = SimpleITK.GetArrayFromImage(resampled_sitk_image)
    return resampled_array


def resample_image_SimpleITK_DASK(sitk_image: SimpleITK.Image, interpolation: str,
                                  output_spacing: tuple = (1.5, 1.5, 1.5)) -> SimpleITK.Image:
    """
    Resamples a sitk_image using dask and scipy. Uses SimpleITK as IO.
    :param sitk_image: The SimpleITK image to be resampled
    :param interpolation: nearest|linear|bspline
    :param output_spacing: The desired output spacing of the resampled sitk_image
    :return: The resampled sitk_image
    :rtype: SimpleITK.Image
    """

    if interpolation == 'nearest':
        interpolation_method = SimpleITK.sitkNearestNeighbor
    elif interpolation == 'linear':
        interpolation_method = SimpleITK.sitkLinear
    elif interpolation == 'bspline':
        interpolation_method = SimpleITK.sitkBSpline
    else:
        raise ValueError('The interpolation method is not supported.')

    input_spacing = sitk_image.GetSpacing()
    input_size = sitk_image.GetSize()
    input_chunks = (input_size[0] / chunk_along_axis(input_size[0]),
                    input_size[1] / chunk_along_axis(input_size[1]),
                    input_size[2] / chunk_along_axis(input_size[2]))
    input_chunks_reversed = list(reversed(input_chunks))
    image_dask = da.from_array(SimpleITK.GetArrayViewFromImage(sitk_image), chunks=input_chunks_reversed)

    output_chunks = [round(input_chunks[i] * (input_spacing[i] / output_spacing[i])) for i in range(len(input_chunks))]
    output_chunks_reversed = list(reversed(output_chunks))
    result = da.map_blocks(resample_chunk_SimpleITK, image_dask, input_spacing, interpolation_method, output_spacing,
                           chunks=output_chunks_reversed)

    resampled_image = SimpleITK.GetImageFromArray(result)
    resampled_image.SetSpacing(output_spacing)
    resampled_image.SetOrigin(sitk_image.GetOrigin())
    resampled_image.SetDirection(sitk_image.GetDirection())

    return resampled_image


def resample(input_image_path: str, interpolation: str, desired_spacing: list, output_image_path: str = None) -> \
        tuple[Nifti1Image, str | None]:
    """
    Resamples an image to a new spacing.
    :param input_image_path: Path to the input image.
    :param output_image_path: Optional path tAdd missed packageo the output image.
    :param interpolation: The interpolation method to use.
    :param desired_spacing: The new spacing to use.
    :return: The resampled_image and the output_image_path (if provided, else None).
    """

    # Load the image and get necessary information
    input_image = nibabel.load(input_image_path)
    image_data = input_image.get_fdata()
    image_header = input_image.header
    image_affine = input_image.affine
    original_spacing = image_header.get_zooms()
    translation_vector = image_affine[:3, 3]
    rotation_matrix = image_affine[:3, :3]

    # Convert to SimpleITK image format
    image_data_swapped_axes = image_data.swapaxes(0, 2)
    sitk_input_image = SimpleITK.GetImageFromArray(image_data_swapped_axes)
    sitk_input_image.SetSpacing([spacing.item() for spacing in original_spacing])
    axis_flip_matrix = np.diag([-1, -1, 1])
    sitk_input_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
    sitk_input_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).ravel())

    # Interpolation:
    resampled_sitk_image = resample_image_SimpleITK_DASK(sitk_input_image, interpolation, tuple(desired_spacing))
    new_size = resampled_sitk_image.GetSize()

    # Save the resampled image to disk
    # Edit affine to fit new image
    new_affine = image_affine
    for diagonal, spacing in enumerate(desired_spacing):
        new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
            new_affine[diagonal, diagonal])) * spacing

    # Edit header to fit new image
    image_header['pixdim'][1:4] = desired_spacing
    image_header['dim'][1:4] = new_size
    resampled_image = nibabel.Nifti1Image(SimpleITK.GetArrayFromImage(resampled_sitk_image).swapaxes(0, 2),
                                          affine=new_affine,
                                          header=image_header)

    if output_image_path is not None:
        nibabel.save(resampled_image, output_image_path)

    return resampled_image, output_image_path


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
