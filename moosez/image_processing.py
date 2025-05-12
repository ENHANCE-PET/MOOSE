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
import itertools
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import nibabel
import os
import math
from typing import Union, Tuple, List, Dict
from moosez.constants import CHUNK_THRESHOLD_RESAMPLING, CHUNK_THRESHOLD_INFERRING
from moosez import models
from moosez import system


def get_intensity_statistics(image: SimpleITK.Image, mask_image: SimpleITK.Image, model: models.Model, out_csv: str) -> None:
    """
    Get the intensity statistics of a NIFTI image file.

    :param image: The source image from which the intensity statistics are calculated.
    :type image: sitk.Image
    :param mask_image: The multilabel mask image.
    :type mask_image: sitk.Image
    :param model: The model.
    :type model: Model
    :param out_csv: The path to the output CSV file.
    :type out_csv: str
    :return: None
    """
    intensity_statistics = SimpleITK.LabelIntensityStatisticsImageFilter()
    min_intensity = SimpleITK.GetArrayViewFromImage(image).min()
    max_intensity = SimpleITK.GetArrayViewFromImage(image).max()
    bins = int(max_intensity - min_intensity)
    intensity_statistics.SetNumberOfBins(bins)
    intensity_statistics.Execute(mask_image, image)
    stats_list = [(intensity_statistics.GetMean(i), intensity_statistics.GetStandardDeviation(i),
                   intensity_statistics.GetMedian(i), intensity_statistics.GetMaximum(i),
                   intensity_statistics.GetMinimum(i)) for i in intensity_statistics.GetLabels()]
    columns = ['Mean', 'Standard-Deviation', 'Median', 'Maximum', 'Minimum']
    stats_df = pd.DataFrame(data=stats_list, index=intensity_statistics.GetLabels(), columns=columns)
    labels_present = stats_df.index.to_list()
    regions_present = []
    organ_indices_dict = model.organ_indices
    for label in labels_present:
        if label in organ_indices_dict:
            regions_present.append(organ_indices_dict[label])
        else:
            continue
    stats_df.insert(0, 'Regions-Present', np.array(regions_present))
    stats_df.to_csv(out_csv)


def get_shape_statistics(mask_image: SimpleITK.Image, model: models.Model, out_csv: str) -> None:
    """
    Get the shape statistics of a NIFTI image file.

    :param mask_image: The multilabel mask image.
    :type mask_image: sitk.Image
    :param model: The model.
    :type model: Model
    :param out_csv: The path to the output CSV file.
    :type out_csv: str
    :return: None
    """
    label_shape_filter = SimpleITK.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask_image)

    stats_list = [(label_shape_filter.GetPhysicalSize(i),) for i in label_shape_filter.GetLabels() if
                  i != 0]  # exclude background label
    columns = ['Volume(mm3)']
    stats_df = pd.DataFrame(data=stats_list, index=[i for i in label_shape_filter.GetLabels() if i != 0],
                            columns=columns)

    labels_present = stats_df.index.to_list()
    regions_present = []
    organ_indices_dict = model.organ_indices
    for label in labels_present:
        if label in organ_indices_dict:
            regions_present.append(organ_indices_dict[label])
        else:
            continue
    stats_df.insert(0, 'Regions-Present', np.array(regions_present))
    stats_df.to_csv(out_csv)


def limit_fov(image_array: np.array, segmentation_array: np.array, fov_label: Union[List[int], int], largest_component_only: bool = False):

    if largest_component_only:
        segmentation_array = largest_connected_component(segmentation_array, fov_label)

    if isinstance(fov_label, list):
        z_indices = np.where((segmentation_array >= fov_label[0]) & (segmentation_array <= fov_label[1]))[0]
    else:
        z_indices = np.where(segmentation_array == fov_label)[0]
    z_min, z_max = np.min(z_indices), np.max(z_indices)

    # Crop the CT data along the z-axis
    limited_fov_array = image_array[z_min:z_max + 1, :, :]

    return limited_fov_array, {"z_min": z_min, "z_max": z_max, "original_shape": image_array.shape}


def expand_segmentation_fov(limited_fov_segmentation_array: np.ndarray, original_fov_info: Dict) -> np.ndarray:
    z_min = original_fov_info["z_min"]
    z_max = original_fov_info["z_max"]
    original_shape = original_fov_info["original_shape"]
    # Initialize an array of zeros with the shape of the original CT
    filled_segmentation_array = np.zeros(original_shape, np.uint8)
    # Place the cropped segmentation back into its original position
    filled_segmentation_array[z_min:z_max + 1, :, :] = limited_fov_segmentation_array

    return filled_segmentation_array


def largest_connected_component(segmentation_array, intensities):
    """
    Extracts the largest connected component for one or more specific intensities from a multilabel segmentation array
    and returns a new multilabel array where the largest components retain their original intensity.

    Parameters:
    - segmentation_array: 3D or 2D numpy array with multiple labels.
    - intensities: A single intensity or a list of intensities for which the largest component(s) should be extracted.

    Returns:
    - largest_components_multilabel: A multilabel array of the same shape as `segmentation_array`, where the largest
      connected component(s) of the specified intensity or intensities retain their original intensity, and all other
      areas are 0.
    """

    # Ensure intensities is a list (even if only one intensity is provided)
    if not isinstance(intensities, (list, tuple, np.ndarray)):
        intensities = [intensities]

    # Initialize an array to store the largest connected components
    largest_components_multilabel = np.zeros_like(segmentation_array, dtype=segmentation_array.dtype)

    # Loop over each intensity
    for intensity in intensities:
        # Create a binary mask for the current intensity
        binary_mask = segmentation_array == intensity

        # Label connected components in the binary mask
        labeled_array, num_features = ndimage.label(binary_mask)

        # Find the sizes of each connected component
        component_sizes = np.bincount(labeled_array.ravel())

        # Ignore the background (component 0)
        component_sizes[0] = 0

        # Find the largest connected component for this intensity
        largest_component_label = component_sizes.argmax()

        # Create a mask for the largest connected component of this intensity
        largest_component = labeled_array == largest_component_label

        # Assign the original intensity value to the largest connected component
        largest_components_multilabel[largest_component] = intensity

    return largest_components_multilabel


class ImageChunker:
    @staticmethod
    def __compute_interior_indices(axis_length: int, number_of_chunks: int) -> Tuple[List[int], List[int]]:
        chunk_base_size = axis_length // number_of_chunks
        remaining_slices = axis_length % number_of_chunks

        start_indices = []
        end_indices = []
        chunk_current_index = 0

        for i in range(number_of_chunks):
            chunk_size = chunk_base_size + (1 if i < remaining_slices else 0)
            start_indices.append(chunk_current_index)
            chunk_current_index += chunk_size
            end_indices.append(chunk_current_index)

        return start_indices, end_indices

    @staticmethod
    def __chunk_array_with_overlap(array_shape: Union[List[int], Tuple[int, ...]], splits_per_dimension: Union[List[int], Tuple[int, ...]], overlap_per_dimension: Union[List[int], Tuple[int, ...]]) -> List[Dict]:
        dims = array_shape
        num_dims = len(array_shape)
        starts_list = []
        ends_list = []

        for dimension_index in range(num_dims):
            axis_length = dims[dimension_index]
            number_of_chunks = splits_per_dimension[dimension_index]
            start_index, end_index = ImageChunker.__compute_interior_indices(axis_length, number_of_chunks)
            starts_list.append(start_index)
            ends_list.append(end_index)

        chunk_info = []
        for idx in itertools.product(*(range(len(s)) for s in starts_list)):
            chunk_slice = []
            interior_slice = []
            dest_slice = []
            for dimension_index, chunk_index in enumerate(idx):
                start_index = starts_list[dimension_index]
                end_index = ends_list[dimension_index]
                axis_length = dims[dimension_index]
                number_of_chunks = splits_per_dimension[dimension_index]
                overlap = overlap_per_dimension[dimension_index]

                start = max(0, start_index[chunk_index] - overlap if chunk_index > 0 else start_index[chunk_index])
                end = min(axis_length, end_index[chunk_index] + overlap if chunk_index < number_of_chunks - 1 else end_index[chunk_index])

                start_in_chunk = start_index[chunk_index] - start
                end_in_chunk = start_in_chunk + (end_index[chunk_index] - start_index[chunk_index])

                start_in_full = start_index[chunk_index]
                end_in_full = end_index[chunk_index]

                chunk_slice.append(slice(start, end))
                interior_slice.append(slice(start_in_chunk, end_in_chunk))
                dest_slice.append(slice(start_in_full, end_in_full))

            chunk_info.append({
                'chunk_slice': tuple(chunk_slice),
                'interior_slice': tuple(interior_slice),
                'dest_slice': tuple(dest_slice)
            })

        return chunk_info

    @staticmethod
    def array_to_chunks(image_array: np.ndarray, splits_per_dimension: Union[List[int], Tuple[int, ...]], overlap_per_dimension: Union[List[int], Tuple[int, ...]]) -> Tuple[List[np.ndarray], List[Dict]]:
        chunk_info = ImageChunker.__chunk_array_with_overlap(image_array.shape, splits_per_dimension, overlap_per_dimension)
        image_chunks = []
        positions = []

        for info in chunk_info:
            image_chunk = image_array[info['chunk_slice']]
            positions.append({
                'interior_slice': info['interior_slice'],
                'dest_slice': info['dest_slice']
            })
            image_chunks.append(image_chunk)

        return image_chunks, positions

    @staticmethod
    def chunks_to_array(image_chunks: List[np.ndarray], image_chunk_positions: List[Dict], final_shape: Union[List[int], Tuple[int, ...]]) -> np.ndarray:
        final_arr = np.empty(final_shape, dtype=image_chunks[0].dtype)
        for image_chunk, image_chunk_position in zip(image_chunks, image_chunk_positions):
            interior_region = image_chunk[image_chunk_position['interior_slice']]
            final_arr[image_chunk_position['dest_slice']] = interior_region

        return final_arr

    @staticmethod
    def determine_splits(image_array: np.ndarray) -> Tuple:
        image_shape = image_array.shape
        splits = []
        for axis in image_shape:
            if axis == 1:
                split = 1
            else:
                split = max(1, math.ceil(axis / CHUNK_THRESHOLD_INFERRING))
            splits.append(split)

        return tuple(splits)


class ImageResampler:
    @staticmethod
    def chunk_along_axis(axis: int) -> int:
        """
        Determines the maximum number of evenly-sized chunks that the axis can be split into.
        Each chunk is at least of size CHUNK_THRESHOLD.

        :param axis: Length of the axis.
        :type axis: int
        :return: The maximum number of evenly-sized chunks.
        :rtype: int
        :raises ValueError: If axis is negative or if CHUNK_THRESHOLD is less than or equal to 0.
        """
        # Check for negative input values
        if axis < 0:
            raise ValueError('Axis must be non-negative')

        if CHUNK_THRESHOLD_RESAMPLING <= 0:
            raise ValueError('CHUNK_THRESHOLD must be greater than 0')

        # If the axis is smaller than the threshold, it cannot be split into smaller chunks
        if axis < CHUNK_THRESHOLD_RESAMPLING:
            return 1

        # Determine the maximum number of chunks that the axis can be split into
        split = axis // CHUNK_THRESHOLD_RESAMPLING

        # Reduce the number of chunks until the axis is evenly divisible by split
        while axis % split != 0:
            split -= 1

        return split

    @staticmethod
    def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: Tuple, interpolation_method: int,
                                 output_spacing: Tuple, output_size: Tuple) -> da.array:
        """
        Resamples a dask array chunk.

        :param image_chunk: The chunk (part of an image) to be resampled.
        :type image_chunk: da.array
        :param input_spacing: The original spacing of the chunk (part of an image).
        :type input_spacing: tuple
        :param interpolation_method: SimpleITK interpolation type.
        :type interpolation_method: int
        :param output_spacing: Spacing of the newly resampled chunk.
        :type output_spacing: tuple
        :param output_size: Size of the newly resampled chunk.
        :type output_size: tuple
        :return: The resampled chunk (part of an image).
        :rtype: da.array
        """
        sitk_image_chunk = SimpleITK.GetImageFromArray(image_chunk)
        sitk_image_chunk.SetSpacing(input_spacing)

        resampled_sitk_image = SimpleITK.Resample(sitk_image_chunk, output_size, SimpleITK.Transform(),
                                                  interpolation_method, sitk_image_chunk.GetOrigin(), output_spacing,
                                                  sitk_image_chunk.GetDirection(), 0.0,
                                                  sitk_image_chunk.GetPixelIDValue())

        resampled_array = SimpleITK.GetArrayFromImage(resampled_sitk_image)
        return resampled_array

    @staticmethod
    def resample_image_SimpleITK_DASK(sitk_image: SimpleITK.Image, interpolation: str,
                                      output_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
                                      output_size: Union[Tuple, None] = None) -> SimpleITK.Image:
        """
        Resamples a sitk_image using Dask and SimpleITK.

        :param sitk_image: The SimpleITK image to be resampled.
        :type sitk_image: sitk.Image
        :param interpolation: nearest|linear|bspline.
        :type interpolation: str
        :param output_spacing: The desired output spacing of the resampled sitk_image.
        :type output_spacing: tuple
        :param output_size: The new size to use.
        :type output_size: tuple
        :return: The resampled sitk_image as SimpleITK.Image.
        :rtype: sitk.Image
        :raises ValueError: If the interpolation method is not supported.
        """

        resample_result = ImageResampler.resample_image_SimpleITK_DASK_array(sitk_image, interpolation, output_spacing, output_size)

        resampled_image = SimpleITK.GetImageFromArray(resample_result)
        resampled_image.SetSpacing(output_spacing)
        resampled_image.SetOrigin(sitk_image.GetOrigin())
        resampled_image.SetDirection(sitk_image.GetDirection())

        return resampled_image

    @staticmethod
    def reslice_identity(reference_image: SimpleITK.Image, moving_image: SimpleITK.Image,
                         output_image_path: Union[str, None] = None, is_label_image: bool = False) -> SimpleITK.Image:
        """
        Reslices an image to the same space as another image.

        :param reference_image: The reference image.
        :type reference_image: SimpleITK.Image
        :param moving_image: The image to reslice to the reference image.
        :type moving_image: SimpleITK.Image
        :param output_image_path: Path to the resliced image. Default is None.
        :type output_image_path: str
        :param is_label_image: Determines if the image is a label image. Default is False.
        :type is_label_image: bool
        :return: The resliced image as SimpleITK.Image.
        :rtype: SimpleITK.Image
        """
        resampler = SimpleITK.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)

        if is_label_image:
            resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(SimpleITK.sitkLinear)

        resampled_image = resampler.Execute(moving_image)
        resampled_image = SimpleITK.Cast(resampled_image, SimpleITK.sitkInt32)
        if output_image_path is not None:
            SimpleITK.WriteImage(resampled_image, output_image_path)
        return resampled_image

    @staticmethod
    def resample_image_SimpleITK_DASK_array(sitk_image: SimpleITK.Image, interpolation: str,
                                            output_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
                                            output_size: Union[Tuple[float, float, float], None] = None) -> np.array:
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
        input_chunks = [axis / ImageResampler.chunk_along_axis(axis) for axis in input_size]
        input_chunks_reversed = list(reversed(input_chunks))

        image_dask = da.from_array(SimpleITK.GetArrayViewFromImage(sitk_image), chunks=input_chunks_reversed)

        if output_size is not None:
            output_spacing = [input_spacing[i] * (input_size[i] / output_size[i]) for i in range(len(input_size))]

        output_chunks = [round(input_chunks[i] * (input_spacing[i] / output_spacing[i])) for i in
                         range(len(input_chunks))]
        output_chunks_reversed = list(reversed(output_chunks))

        result = da.map_blocks(ImageResampler.resample_chunk_SimpleITK, image_dask, input_spacing, interpolation_method,
                               output_spacing, output_chunks, chunks=output_chunks_reversed, meta=np.array(()),
                               dtype=np.float32)

        return result.compute()

    @staticmethod
    def resample_segmentation(reference_image: SimpleITK.Image, segmentation_image: SimpleITK.Image):
        resampled_sitk_image = SimpleITK.Resample(segmentation_image, reference_image.GetSize(), SimpleITK.Transform(),
                                                  SimpleITK.sitkNearestNeighbor, reference_image.GetOrigin(),
                                                  reference_image.GetSpacing(), reference_image.GetDirection(), 0.0,
                                                  segmentation_image.GetPixelIDValue())
        return resampled_sitk_image


def determine_orientation_code(image: nibabel.Nifti1Image) -> Tuple[Union[Tuple, List], str]:
    affine = image.affine
    orthonormal_orientation = nibabel.orientations.aff2axcodes(affine)
    return orthonormal_orientation, ''.join(orthonormal_orientation)


def confirm_orthonormality(image: nibabel.Nifti1Image) -> Tuple[nibabel.Nifti1Image, bool]:
    data = image.get_fdata()
    affine = image.affine
    header = image.header

    rotation_matrix = affine[:3, :3]
    spacing = np.linalg.norm(rotation_matrix, axis=0)

    ortho_rotation_matrix = rotation_matrix / spacing
    is_orthonormal = np.allclose(ortho_rotation_matrix.T @ ortho_rotation_matrix, np.eye(3))

    if not is_orthonormal:
        orthonormalized = True
        q, _ = np.linalg.qr(ortho_rotation_matrix)
        ortho_rotation_matrix = q * spacing

        orthonormal_affine = np.eye(4)
        orthonormal_affine[:3, :3] = ortho_rotation_matrix
        orthonormal_affine[:3, 3] = affine[:3, 3]

        orthonormal_header = header.copy()
        orthonormal_header.set_qform(orthonormal_affine)
        orthonormal_header.set_sform(orthonormal_affine)

        image = nibabel.Nifti1Image(data, orthonormal_affine, orthonormal_header)
    else:
        orthonormalized = False

    return image, orthonormalized


def confirm_orientation(image: nibabel.Nifti1Image) -> Tuple[nibabel.Nifti1Image, bool]:
    data = image.get_fdata()
    affine = image.affine
    header = image.header

    original_orientation = nibabel.orientations.aff2axcodes(affine)

    if original_orientation[0] == 'R':
        reoriented = True

        current_orientation = nibabel.orientations.axcodes2ornt(original_orientation)
        target_orientation = nibabel.orientations.axcodes2ornt(('L', original_orientation[1], original_orientation[2]))

        orientation_transform = nibabel.orientations.ornt_transform(current_orientation, target_orientation)
        reoriented_data = nibabel.orientations.apply_orientation(data, orientation_transform)
        reoriented_affine = nibabel.orientations.inv_ornt_aff(orientation_transform, data.shape).dot(affine)

        reoriented_header = header.copy()
        reoriented_header.set_qform(reoriented_affine)
        reoriented_header.set_sform(reoriented_affine)

        image = nibabel.Nifti1Image(reoriented_data, reoriented_affine, reoriented_header)
    else:
        reoriented = False

    return image, reoriented


def convert_to_sitk(image: nibabel.Nifti1Image) -> SimpleITK.Image:
    data = image.get_fdata()
    affine = image.affine
    spacing = image.header.get_zooms()

    image_data_swapped_axes = data.swapaxes(0, 2)
    sitk_image = SimpleITK.GetImageFromArray(image_data_swapped_axes)

    translation_vector = affine[:3, 3]
    rotation_matrix = affine[:3, :3]
    axis_flip_matrix = np.diag([-1, -1, 1])

    sitk_image.SetSpacing([spacing.item() for spacing in spacing])
    sitk_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
    sitk_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(spacing)).flatten())

    return sitk_image


def standardize_image(image_path: str, output_manager: system.OutputManager, standardization_output_path: Union[str, None]) -> SimpleITK.Image:
    image = nibabel.load(image_path)
    _, original_orientation = determine_orientation_code(image)
    output_manager.log_update(f" - Image loaded. Orientation: {original_orientation}")

    image, orthonormalized = confirm_orthonormality(image)
    if orthonormalized:
        _, orthonormal_orientation = determine_orientation_code(image)
        output_manager.log_update(f"   - Image orthonormalized. Orientation: {orthonormal_orientation}")
    image, reoriented = confirm_orientation(image)
    if reoriented:
        _, reoriented_orientation = determine_orientation_code(image)
        output_manager.log_update(f"   - Image reoriented. Orientation: {reoriented_orientation}")
    sitk_image = convert_to_sitk(image)
    output_manager.log_update(f" - Image converted to SimpleITK.")

    processing_steps = [orthonormalized, reoriented]
    prefixes = ["orthonormal", "reoriented"]

    if standardization_output_path is not None and any(processing_steps):
        output_manager.log_update(f" - Writing standardized image.")
        prefix = "_".join([prefix for processing_step, prefix in zip(processing_steps, prefixes) if processing_step])
        output_path = os.path.join(standardization_output_path, f"{prefix}_{os.path.basename(image_path)}")
        SimpleITK.WriteImage(sitk_image, output_path)

    return sitk_image