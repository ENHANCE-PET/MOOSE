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
from typing import Optional


def chunk_along_axis(axis: int) -> int:
    if axis < constants.CHUNK_THRESHOLD:
        return axis

    split = 2
    result = axis // split
    rest = axis % split

    while result >= constants.CHUNK_THRESHOLD or rest != 0:
        split += 1
        result = axis // split
        rest = axis % split
        if split == axis:
            return 1

    return split


def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: tuple, interpolation_method: str,
                             output_spacing: tuple = (1.5, 1.5, 1.5), output_size: tuple = None) -> da.array:
    sitk_image_chunk = SimpleITK.GetImageFromArray(image_chunk)
    sitk_image_chunk.SetSpacing(input_spacing)
    input_size = sitk_image_chunk.GetSize()

    if output_size is None:
        output_size = [max(1, round(input_size[i] * (input_spacing[i] / max(1e-3, output_spacing[i])))) for i in
                       range(len(input_size))]
    else:
        output_spacing = [max(1e-3, input_spacing[i] * (input_size[i] / max(1, output_size[i]))) for i in
                          range(len(input_size))]

    if all(x == 0 for x in input_size):
        return image_chunk

    resampled_sitk_image = SimpleITK.Resample(sitk_image_chunk, output_size, SimpleITK.Transform(),
                                              interpolation_method,
                                              sitk_image_chunk.GetOrigin(), output_spacing,
                                              sitk_image_chunk.GetDirection(), 0.0, sitk_image_chunk.GetPixelIDValue())

    resampled_array = SimpleITK.GetArrayFromImage(resampled_sitk_image)
    return resampled_array


def resample_image_SimpleITK_DASK(sitk_image: SimpleITK.Image, interpolation: str,
                                  output_spacing: tuple = (1.5, 1.5, 1.5),
                                  output_size: tuple = None) -> SimpleITK.Image:
    # Interpolation method
    if interpolation == 'nearest':
        interpolation_method = SimpleITK.sitkNearestNeighbor
    elif interpolation == 'linear':
        interpolation_method = SimpleITK.sitkLinear
    elif interpolation == 'bspline':
        interpolation_method = SimpleITK.sitkBSpline
    else:
        raise ValueError('The interpolation method is not supported.')

    # Calculate input spacing and size
    input_spacing = sitk_image.GetSpacing()
    input_size = sitk_image.GetSize()

    # Calculate chunks
    input_chunks = [max(1, input_size[i] // chunk_along_axis(input_size[i])) for i in range(len(input_size))]
    input_chunks_reversed = list(reversed(input_chunks))
    image_dask = da.from_array(SimpleITK.GetArrayViewFromImage(sitk_image), chunks=input_chunks_reversed)

    # Calculate output spacing if output size is provided
    if output_size is not None:
        output_spacing = [max(1e-3, input_spacing[i] * (input_size[i] / max(1, output_size[i]))) for i in
                          range(len(input_size))]

    # Calculate output chunks
    output_chunks = [max(1, round(input_chunks[i] * (input_spacing[i] / max(1e-3, output_spacing[i])))) for i in
                     range(len(input_chunks))]
    output_chunks_reversed = list(reversed(output_chunks))

    result = da.map_blocks(resample_chunk_SimpleITK, image_dask, input_spacing, interpolation_method, output_spacing,
                           output_size, chunks=output_chunks_reversed)

    resampled_image = SimpleITK.GetImageFromArray(result)
    resampled_image.SetSpacing(output_spacing)
    resampled_image.SetOrigin(sitk_image.GetOrigin())
    resampled_image.SetDirection(sitk_image.GetDirection())

    return resampled_image


def resample(input_image_path: str, interpolation: str, desired_spacing: list, output_image_path: str = None,
             desired_size: list = None) -> tuple[Nifti1Image, str]:
    input_image = nibabel.load(input_image_path)
    image_data = input_image.get_fdata()
    image_header = input_image.header
    image_affine = input_image.affine
    original_spacing = image_header.get_zooms()
    translation_vector = image_affine[:3, 3]
    rotation_matrix = image_affine[:3, :3]

    image_data_swapped_axes = image_data.swapaxes(0, 2)
    sitk_input_image = SimpleITK.GetImageFromArray(image_data_swapped_axes)
    sitk_input_image.SetSpacing([spacing.item() for spacing in original_spacing])
    axis_flip_matrix = np.diag([-1, -1, 1])
    sitk_input_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
    sitk_input_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).ravel())

    resampled_sitk_image = resample_image_SimpleITK_DASK(sitk_input_image, interpolation, tuple(desired_spacing),
                                                         output_size=desired_size)
    if desired_size is None:
        new_size = resampled_sitk_image.GetSize()
    else:
        new_size = desired_size

    new_affine = image_affine
    for diagonal, spacing in enumerate(desired_spacing):
        new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
            new_affine[diagonal, diagonal])) * spacing

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


def get_size(image_path):
    """
    Returns the size of an image.
    :param image_path: Path to the image.
    :return: The size of the image.
    """
    image = nibabel.load(image_path)
    image_header = image.header
    size = image_header.get_data_shape()
    return [dimension for dimension in size]


def resample_mask(image_path, mask_path, output_mask_path, interpolation_method):
    # Load the image and mask
    image = SimpleITK.ReadImage(image_path)
    mask = SimpleITK.ReadImage(mask_path)

    # Create resampler
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(image)  # resample the mask to match the image
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetSize(image.GetSize())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(SimpleITK.Transform())  # identity transform

    # Set interpolation method
    if interpolation_method == 'nearest':
        resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    elif interpolation_method == 'linear':
        resampler.SetInterpolator(SimpleITK.sitkLinear)
    else:
        raise ValueError('The interpolation method is not supported.')

    # Resample the mask
    resampled_mask = resampler.Execute(mask)

    # Write the output image
    SimpleITK.WriteImage(resampled_mask, output_mask_path)

    return resampled_mask



def resample_image_SimpleITK(input_image_path: str, interpolation: str, output_spacing: tuple = (1.5, 1.5, 1.5),
                             output_image_path: Optional[str] = None, output_size: Optional[tuple] = None) -> tuple[Nifti1Image, str]:
    # Load the image
    input_image = nibabel.load(input_image_path)
    image_data = input_image.get_fdata()
    image_header = input_image.header
    image_affine = input_image.affine
    original_spacing = image_header.get_zooms()
    translation_vector = image_affine[:3, 3]
    rotation_matrix = image_affine[:3, :3]

    # Convert nibabel image to SimpleITK image for resampling
    image_data_swapped_axes = image_data.swapaxes(0, 2)
    sitk_image = SimpleITK.GetImageFromArray(image_data_swapped_axes)
    sitk_image.SetSpacing([spacing.item() for spacing in original_spacing])
    axis_flip_matrix = np.diag([-1, -1, 1])
    sitk_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
    sitk_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).ravel())

    # Interpolation method
    if interpolation == 'nearest':
        interpolation_method = SimpleITK.sitkNearestNeighbor
    elif interpolation == 'linear':
        interpolation_method = SimpleITK.sitkLinear
    elif interpolation == 'bspline':
        interpolation_method = SimpleITK.sitkBSpline
    else:
        raise ValueError('The interpolation method is not supported.')

    # Set up the resampler
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(output_size if output_size is not None else sitk_image.GetSize())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(SimpleITK.Transform())  # identity transform
    resampler.SetInterpolator(interpolation_method)

    # Apply the resampling
    resampled_sitk_image = resampler.Execute(sitk_image)

    # Convert the resampled image back to nibabel format
    new_size = resampled_sitk_image.GetSize()
    new_affine = image_affine.copy()
    for diagonal, spacing in enumerate(output_spacing):
        new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
            new_affine[diagonal, diagonal])) * spacing
    image_header['pixdim'][1:4] = output_spacing
    image_header['dim'][1:4] = new_size
    resampled_image = nibabel.Nifti1Image(SimpleITK.GetArrayFromImage(resampled_sitk_image).swapaxes(0, 2),
                                          affine=new_affine,
                                          header=image_header)

    # Save the resampled image if a path was provided
    if output_image_path is not None:
        nibabel.save(resampled_image, output_image_path)

    return resampled_image, output_image_path




#---------------------------------#
#  SPRINT SESSION: 1 <26.06.2023> #
#---------------------------------#

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import constants

class MooseObject:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.large_image = False
        if image_path is not None:
            self.load_image()

    def load_image(self):
        # Load as nibabel image
        self.nib_image = nib.load(self.image_path)
        self.original_properties = {'affine': self.nib_image.affine, 
                                    'header': self.nib_image.header, 
                                    'shape': self.nib_image.shape}
        # Convert nibabel image to SimpleITK image
        self.image = self.nib_to_sitk(self.nib_image)

    def nib_to_sitk(self, nib_image):
        data_array = nib_image.get_fdata()
        image = sitk.GetImageFromArray(data_array)
        affine = nib_image.affine
        image.SetOrigin(affine[:3,3])
        image.SetDirection(affine[:3,:3].flatten())
        return image

    def save_image(self, output_path):
        # Save SimpleITK image
        sitk.WriteImage(self.image, output_path)

    def remove_singleton_dims(self):
        self.nib_image = nib.squeeze_image(self.nib_image)

    def standardize_orientation(self):
        if nib.aff2axcodes(self.nib_image.affine) != ('L', 'A', 'S'):
            self.nib_image = nib.as_reoriented(self.nib_image, nib.aff2axcodes(self.nib_image.affine), ('L', 'A', 'S'))

    def orient_image(self):
        self.nib_image = nib.as_closest_canonical(self.nib_image)

    def check_size(self):
        total_voxels = np.prod(self.nib_image.shape)
        z_axis_length = self.nib_image.shape[2]
        if total_voxels > constants.MATRIX_THRESHOLD and z_axis_length > constants.Z_AXIS_THRESHOLD:
            self.large_image = True

    def get_original_properties(self):
        return self.original_properties

    def create_image_from_scratch(self, data, affine=np.eye(4)):
        self.nib_image = nib.Nifti1Image(data, affine)
        self.image = self.nib_to_sitk(self.nib_image)

    def run_preprocessing_pipeline(self):
        self.remove_singleton_dims()
        self.standardize_orientation()
        self.orient_image()
        self.check_size()
        # Convert preprocessed nibabel image to SimpleITK image
        self.image = self.nib_to_sitk(self.nib_image)

    def is_large_image(self):
        return self.large_image

