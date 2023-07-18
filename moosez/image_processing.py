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

import os
import dask
import SimpleITK as sitk
import dask.array as da
import nibabel
import numpy as np
import pandas as pd

from moosez.constants import MATRIX_THRESHOLD, Z_AXIS_THRESHOLD, CHUNK_THRESHOLD, MARGIN_PADDING, ORGAN_INDICES, \
    CHUNK_FILENAMES


def get_intensity_statistics(image: sitk.Image, mask_image: sitk.Image, model_name: str, out_csv: str) -> None:
    """
    Get the intensity statistics of a NIFTI image file
    :param image: Source image from which the intensity statistics are calculated
    :param mask_image: Multilabel mask image
    :param model_name: Name of the model
    :param out_csv: Path to the output csv file
    :return None
     """
    intensity_statistics = sitk.LabelIntensityStatisticsImageFilter()
    intensity_statistics.Execute(mask_image, image)
    stats_list = [(intensity_statistics.GetMean(i), intensity_statistics.GetStandardDeviation(i),
                   intensity_statistics.GetMedian(i), intensity_statistics.GetMaximum(i),
                   intensity_statistics.GetMinimum(i)) for i in intensity_statistics.GetLabels()]
    columns = ['Mean', 'Standard-Deviation', 'Median', 'Maximum', 'Minimum']
    stats_df = pd.DataFrame(data=stats_list, index=intensity_statistics.GetLabels(), columns=columns)
    labels_present = stats_df.index.to_list()
    regions_present = []
    organ_indices_dict = ORGAN_INDICES[model_name]
    for label in labels_present:
        if label in organ_indices_dict:
            regions_present.append(organ_indices_dict[label])
        else:
            continue
    stats_df.insert(0, 'Regions-Present', np.array(regions_present))
    stats_df.to_csv(out_csv)


def split_and_save(image_chunk: da.Array, image_affine, image_chunk_path: str) -> None:
    """
    Get a chunk of the image and save it.

    Args:
        image_chunk: Dask array chunk.
        image_affine: Image affine transformation.
        image_chunk_path: The path to save the image chunk.
    """
    chunk_part = nibabel.Nifti1Image(image_chunk, image_affine)
    nibabel.save(chunk_part, image_chunk_path)


@dask.delayed
def delayed_split_and_save(image_chunk, image_affine, image_chunk_path):
    split_and_save(image_chunk, image_affine, image_chunk_path)


def write_image(image: nibabel.Nifti1Image, out_image_path: str, large_image: bool = False, is_label: bool = False) -> None:
    """
    Writes an image either as a single file or multiple files depending on the image size.

    Args:
        image: The image to save.
        out_image_path: The path to save the image.
        large_image: Indicates whether the image classifies as large or not.
        is_label: Indicates whether the image is a label or not.
    """
    image_shape = image.shape
    image_data = da.from_array(image.get_fdata(), chunks=(image_shape[0], image_shape[1], image_shape[2] // 3))
    image_affine = image.affine

    if large_image:
        # Calculate indices for image chunks
        num_chunks = 3
        chunk_size = image_shape[2] // num_chunks
        chunk_indices = [(0, chunk_size + MARGIN_PADDING),
                         (chunk_size + 1 - MARGIN_PADDING, chunk_size * 2 + MARGIN_PADDING),
                         (chunk_size * 2 + 1 - MARGIN_PADDING, None)]
        filenames = CHUNK_FILENAMES
        save_dir = os.path.dirname(out_image_path)
        chunk_paths = [os.path.join(save_dir, filename) for filename in filenames]

        tasks = []
        for i, chunk_path in enumerate(chunk_paths):
            tasks.append(delayed_split_and_save(image_data[:, :, chunk_indices[i][0]:chunk_indices[i][1]].compute(),
                                                image_affine, chunk_path))

        dask.compute(*tasks)

    else:
        if is_label:
            resampled_image_path = out_image_path
            image_as_uint8 = nibabel.Nifti1Image(image.get_fdata().astype(np.uint8), image.affine)
            nibabel.save(image_as_uint8, resampled_image_path)
        else:
            resampled_image_path = out_image_path
            nibabel.save(image, resampled_image_path)


class NiftiPreprocessor:
    """
    A class for processing NIfTI images using nibabel and SimpleITK.

    Attributes:
    -----------
    image: nibabel.Nifti1Image
        The NIfTI image to be processed.
    original_header: nibabel Header
        The original header information of the NIfTI image.
    is_large: bool
        Flag indicating if the image is classified as large.
    sitk_image: SimpleITK.Image
        The image converted into a SimpleITK object.
    """

    def __init__(self, image: nibabel.Nifti1Image):
        """
        Constructs all the necessary attributes for the ImageProcessor object.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to be processed.
        """
        self.image = image
        self.original_header = image.header.copy()
        self.is_large = self._is_large_image(image.shape)
        # if not self._is_orthonormal(self.image):
        #     print('Image is not orthonormal. Making it orthonormal.')
        #     self.image = self._make_orthonormal(self.image)
        self.sitk_image = self._convert_to_sitk(self.image)

    @staticmethod
    def _is_large_image(image_shape) -> bool:
        """
        Check if the image classifies as large based on pre-defined thresholds.

        Parameters:
        -----------
        image_shape: tuple
            The shape of the NIfTI image.

        Returns:
        --------
        bool
            True if the image is large, False otherwise.
        """
        return np.prod(image_shape) > MATRIX_THRESHOLD and image_shape[2] > Z_AXIS_THRESHOLD

    @staticmethod
    def _is_orthonormal(image: nibabel.Nifti1Image) -> bool:
        """
        Check if the qform or sform of a NIfTI image is orthonormal.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to check.

        Returns:
        --------
        bool
            True if qform or sform is orthonormal, False otherwise.
        """
        # Check qform
        qform_code = image.header["qform_code"]
        if qform_code != 0:
            qform = image.get_qform()
            q_rotation = qform[:3, :3]
            q_orthonormal = np.allclose(np.dot(q_rotation, q_rotation.T), np.eye(3))
            # if not q_orthonormal:
            # return False

        # Check sform
        sform = image.get_sform()
        s_rotation = sform[:3, :3]
        s_orthonormal = np.allclose(np.dot(s_rotation, s_rotation.T), np.eye(3))
        if not s_orthonormal:
            return False

        return True

    @staticmethod
    def _make_orthonormal(image: nibabel.Nifti1Image) -> nibabel.Nifti1Image:
        """
        Make a NIFTI image orthonormal while keeping the diagonal of the rotation matrix positive.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to make orthonormal.

        Returns:
        --------
        nibabel.Nifti1Image
            The orthonormal NIFTI image.
        """
        new_affine = image.affine
        new_header = image.header

        rotation_scaling = new_affine[:3, :3]

        q, r = np.linalg.qr(rotation_scaling)
        diagonal_sign = np.sign(np.diag(r))
        q = q @ np.diag(diagonal_sign)
        orthonormal = q

        new_affine[:3, :3] = orthonormal
        new_header['pixdim'][1:4] = np.diag(orthonormal)
        new_header['srow_x'] = new_affine[0, :]
        new_header['srow_y'] = new_affine[1, :]
        new_header['srow_z'] = new_affine[2, :]

        new_image = nibabel.Nifti1Image(image.get_fdata(), affine=new_affine, header=new_header)

        return new_image

    @staticmethod
    def _convert_to_sitk(image: nibabel.Nifti1Image) -> sitk.Image:
        """
        Convert a NIfTI image to a SimpleITK image, retaining the original header information.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to convert.

        Returns:
        --------
        sitk.Image
            The SimpleITK image.
        """
        image_data = image.get_fdata()
        image_affine = image.affine
        original_spacing = image.header.get_zooms()

        image_data_swapped_axes = image_data.swapaxes(0, 2)
        sitk_image = sitk.GetImageFromArray(image_data_swapped_axes)

        translation_vector = image_affine[:3, 3]
        rotation_matrix = image_affine[:3, :3]
        axis_flip_matrix = np.diag([-1, -1, 1])

        sitk_image.SetSpacing([spacing.item() for spacing in original_spacing])
        sitk_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
        sitk_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).flatten())

        return sitk_image


class ImageResampler:
    @staticmethod
    def chunk_along_axis(axis: int) -> int:
        """
        Determines the maximum number of evenly-sized chunks that the axis can be split into.
        Each chunk is at least of size CHUNK_THRESHOLD.

        Args:
            axis (int): Length of the axis.

        Returns:
            int: The maximum number of evenly-sized chunks.

        Raises:
            ValueError: If axis is negative or if CHUNK_THRESHOLD is less than or equal to 0.
        """
        # Check for negative input values
        if axis < 0:
            raise ValueError('Axis must be non-negative')

        if CHUNK_THRESHOLD <= 0:
            raise ValueError('CHUNK_THRESHOLD must be greater than 0')

        # If the axis is smaller than the threshold, it cannot be split into smaller chunks
        if axis < CHUNK_THRESHOLD:
            return 1

        # Determine the maximum number of chunks that the axis can be split into
        split = axis // CHUNK_THRESHOLD

        # Reduce the number of chunks until axis is evenly divisible by split
        while axis % split != 0:
            split -= 1

        return split

    @staticmethod
    def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: tuple, interpolation_method: int,
                                 output_spacing: tuple, output_size: tuple) -> da.array:
        """
        Resamples a dask array chunk.

        Args:
            image_chunk: The chunk (part of an image) to be resampled.
            input_spacing: The original spacing of the chunk (part of an image).
            interpolation_method: SimpleITK interpolation type.
            output_spacing: Spacing of the newly resampled chunk.
            output_size: Size of the newly resampled chunk.

        Returns:
            resampled_array: The resampled chunk (part of an image).
        """
        sitk_image_chunk = sitk.GetImageFromArray(image_chunk)
        sitk_image_chunk.SetSpacing(input_spacing)
        input_size = sitk_image_chunk.GetSize()

        if all(x == 0 for x in input_size):
            return image_chunk

        resampled_sitk_image = sitk.Resample(sitk_image_chunk, output_size, sitk.Transform(),
                                             interpolation_method,
                                             sitk_image_chunk.GetOrigin(), output_spacing,
                                             sitk_image_chunk.GetDirection(), 0.0, sitk_image_chunk.GetPixelIDValue())

        resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)
        return resampled_array

    @staticmethod
    def resample_image_SimpleITK_DASK(sitk_image: sitk.Image, interpolation: str,
                                      output_spacing: tuple = (1.5, 1.5, 1.5),
                                      output_size: tuple = None) -> sitk.Image:
        """
        Resamples a sitk_image using Dask and SimpleITK.

        Args:
            sitk_image: The SimpleITK image to be resampled.
            interpolation: nearest|linear|bspline.
            output_spacing: The desired output spacing of the resampled sitk_image.
            output_size: The new size to use.

        Returns:
            resampled_image: The resampled sitk_image as SimpleITK.Image.
        """
        if interpolation == 'nearest':
            interpolation_method = sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation == 'bspline':
            interpolation_method = sitk.sitkBSpline
        else:
            raise ValueError('The interpolation method is not supported.')

        input_spacing = sitk_image.GetSpacing()
        input_size = sitk_image.GetSize()
        input_chunks = (input_size[0] / ImageResampler.chunk_along_axis(input_size[0]),
                        input_size[1] / ImageResampler.chunk_along_axis(input_size[1]),
                        input_size[2] / ImageResampler.chunk_along_axis(input_size[2]))
        input_chunks_reversed = list(reversed(input_chunks))

        image_dask = da.from_array(sitk.GetArrayViewFromImage(sitk_image), chunks=input_chunks_reversed)

        if output_size is not None:
            output_spacing = [input_spacing[i] * (input_size[i] / output_size[i]) for i in range(len(input_size))]

        output_chunks = [round(input_chunks[i] * (input_spacing[i] / output_spacing[i])) for i in
                         range(len(input_chunks))]
        output_chunks_reversed = list(reversed(output_chunks))

        result = da.map_blocks(ImageResampler.resample_chunk_SimpleITK, image_dask, input_spacing, interpolation_method,
                               output_spacing, output_chunks, chunks=output_chunks_reversed)

        resampled_image = sitk.GetImageFromArray(result)
        resampled_image.SetSpacing(output_spacing)
        resampled_image.SetOrigin(sitk_image.GetOrigin())
        resampled_image.SetDirection(sitk_image.GetDirection())

        return resampled_image

    @staticmethod
    def resample_image_SimpleITK(sitk_image: sitk.Image, interpolation: str,
                                 output_spacing: tuple = (1.5, 1.5, 1.5),
                                 output_size: tuple = None) -> sitk.Image:
        """
        Resamples an image to a new spacing using SimpleITK.

        Args:
            sitk_image: The input image.
            interpolation: nearest | linear | bspline.
            output_size: The new size to use.
            output_spacing: The new spacing to use.

        Returns:
            resampled_image: The resampled image as SimpleITK.Image.
        """
        if interpolation == 'nearest':
            interpolation_method = sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation == 'bspline':
            interpolation_method = sitk.sitkBSpline
        else:
            raise ValueError('The interpolation method is not supported.')

        desired_spacing = np.array(output_spacing).astype(np.float64)
        if output_size is None:
            input_size = sitk_image.GetSize()
            input_spacing = sitk_image.GetSpacing()
            output_size = [round(input_size[i] * (input_spacing[i] / output_spacing[i])) for i in
                           range(len(input_size))]

        # Interpolation:
        resampled_sitk_image = sitk.Resample(sitk_image, output_size, sitk.Transform(), interpolation_method,
                                             sitk_image.GetOrigin(), desired_spacing,
                                             sitk_image.GetDirection(), 0.0, sitk_image.GetPixelIDValue())

        return resampled_sitk_image

    @staticmethod
    def resample_image(moose_img_object, interpolation: str, desired_spacing: tuple,
                       desired_size: tuple = None) -> nibabel.Nifti1Image:
        """
        Resamples an image to a new spacing.

        Args:
            moose_img_object: The moose_img_object to be resampled.
            interpolation: The interpolation method to use.
            desired_spacing: The new spacing to use.
            desired_size: The new size to use.

        Returns:
            resampled_image: The resampled image as nibabel.Nifti1Image.
        """

        image_header = moose_img_object.original_header
        image_affine = moose_img_object.image.affine
        sitk_input_image = moose_img_object.sitk_image
        # Resampling scheme based on image size
        if moose_img_object.is_large:
            resampled_sitk_image = ImageResampler.resample_image_SimpleITK_DASK(sitk_input_image, interpolation,
                                                                                desired_spacing, desired_size)
        else:
            resampled_sitk_image = ImageResampler.resample_image_SimpleITK(sitk_input_image, interpolation,
                                                                           desired_spacing, desired_size)

        new_size = resampled_sitk_image.GetSize()

        # Edit affine to fit the new image
        new_affine = image_affine
        for diagonal, spacing in enumerate(desired_spacing):
            new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
                new_affine[diagonal, diagonal])) * spacing

        # Edit header to fit the new image
        image_header['pixdim'][1:4] = desired_spacing
        image_header['dim'][1:4] = new_size
        image_header['srow_x'] = new_affine[0, :]
        image_header['srow_y'] = new_affine[1, :]
        image_header['srow_z'] = new_affine[2, :]

        resampled_image = nibabel.Nifti1Image(sitk.GetArrayFromImage(resampled_sitk_image).swapaxes(0, 2),
                                              affine=new_affine,
                                              header=image_header)

        return resampled_image

    @staticmethod
    def resample_segmentations(input_image_path: str, desired_spacing: tuple,
                               desired_size: tuple) -> nibabel.Nifti1Image:
        """
        Resamples an image to a new spacing.

        Args:
            input_image_path: Path to the input image.
            desired_spacing: The new spacing to use.
            desired_size: The new size to use.

        Returns:
            resampled_image: The resampled image as nibabel.Nifti1Image.
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
        sitk_input_image = sitk.GetImageFromArray(image_data_swapped_axes)
        sitk_input_image.SetSpacing([spacing.item() for spacing in original_spacing])
        axis_flip_matrix = np.diag([-1, -1, 1])
        sitk_input_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
        sitk_input_image.SetDirection(
            (np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).ravel())

        desired_spacing = np.array(desired_spacing).astype(np.float64)

        # Interpolation:
        resampled_sitk_image = sitk.Resample(sitk_input_image, desired_size, sitk.Transform(),
                                             sitk.sitkNearestNeighbor,
                                             sitk_input_image.GetOrigin(), desired_spacing,
                                             sitk_input_image.GetDirection(), 0.0, sitk_input_image.GetPixelIDValue())

        # Edit affine to fit the new image
        new_affine = image_affine
        for diagonal, spacing in enumerate(desired_spacing):
            new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
                new_affine[diagonal, diagonal])) * spacing

        # Edit header to fit the new image
        image_header['pixdim'][1:4] = desired_spacing
        image_header['dim'][1:4] = desired_size
        image_header['srow_x'] = new_affine[0, :]
        image_header['srow_y'] = new_affine[1, :]
        image_header['srow_z'] = new_affine[2, :]

        resampled_image = nibabel.Nifti1Image(sitk.GetArrayFromImage(resampled_sitk_image).swapaxes(0, 2),
                                              affine=new_affine,
                                              header=image_header)

        return resampled_image

    @staticmethod
    def reslice_identity(reference_image: sitk.Image, moving_image: sitk.Image,
                         output_image_path: str = None, is_label_image: bool = False) -> sitk.Image:
        """
        Reslice an image to the same space as another image
        :param reference_image: The reference image
        :param moving_image: The image to reslice to the reference image
        :param output_image_path: Path to the resliced image
        :param is_label_image: Determines if the image is a label image. Default is False
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)

        if is_label_image:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        resampled_image = resampler.Execute(moving_image)
        resampled_image = sitk.Cast(resampled_image, sitk.sitkInt32)
        if output_image_path is not None:
            sitk.WriteImage(resampled_image, output_image_path)
        return resampled_image
