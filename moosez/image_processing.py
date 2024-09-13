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
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from moosez.constants import CHUNK_THRESHOLD
from moosez.resources import MODELS
from moosez import predict
from moosez import models


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


def limit_fov(image_array: np.array, segmentation_array: np.array, fov_label: list[int] | int, largest_component_only: bool = False):

    if largest_component_only:
        segmentation_array = largest_connected_component(segmentation_array, fov_label)

    if type(fov_label) is list:
        z_indices = np.where((segmentation_array >= fov_label[0]) & (segmentation_array <= fov_label[1]))[0]
    else:
        z_indices = np.where(segmentation_array == fov_label)[0]
    z_min, z_max = np.min(z_indices), np.max(z_indices)

    # Crop the CT data along the z-axis
    limited_fov_array = image_array[z_min:z_max + 1, :, :]

    return limited_fov_array, {"z_min": z_min, "z_max": z_max, "original_shape": image_array.shape}


def expand_segmentation_fov(limited_fov_segmentation_array: np.ndarray, original_fov_info: dict) -> np.ndarray:
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


def cropped_fov_prediction_pipeline(image, segmentation_array, routine: models.ModelSequence, accelerator, nnunet_log_filename):
    """
    Process segmentation by resampling, limiting FOV, and predicting.

    Parameters:
        image (SimpleITK.Image): The input image.
        segmentation_array (np.array): The segmentation array to be processed.
        routine (models.ModelSequence): List of routines where the second element contains model info.
        accelerator (any): The accelerator used for prediction.
        nnunet_log_filename (str): Path to the nnunet log file.

    Returns:
        model (str): The model name used in the process.
        segmentation_array (np.array): The final processed segmentation array.
    """
    # Get the second model from the routine
    model_to_crop_from = routine[0]
    target_model = routine[1]
    target_model_fov_information = target_model.limit_fov

    # Convert the segmentation array to SimpleITK image and set properties
    to_crop_segmentation = SimpleITK.GetImageFromArray(segmentation_array)
    to_crop_segmentation.SetOrigin(image.GetOrigin())
    to_crop_segmentation.SetSpacing(model_to_crop_from.voxel_spacing)
    to_crop_segmentation.SetDirection(image.GetDirection())

    # Resample the image using the desired spacing
    desired_spacing = target_model.voxel_spacing
    to_crop_image_array = ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)
    to_crop_image = SimpleITK.GetImageFromArray(to_crop_image_array)
    to_crop_image.SetOrigin(image.GetOrigin())
    to_crop_image.SetSpacing(desired_spacing)
    to_crop_image.SetDirection(image.GetDirection())

    # Resample the segmentation
    resampled_to_crop_segmentation = ImageResampler.resample_segmentation(to_crop_image, to_crop_segmentation)
    del to_crop_segmentation
    resampled_to_crop_segmentation_array = SimpleITK.GetArrayFromImage(resampled_to_crop_segmentation)

    # Limit FOV based on model information
    limited_fov_image_array, original_fov_info = limit_fov(to_crop_image_array, resampled_to_crop_segmentation_array,
                                                           target_model_fov_information["inference_fov_intensities"])

    to_write_image = SimpleITK.GetImageFromArray(limited_fov_image_array)
    to_write_image.SetOrigin(image.GetOrigin())
    to_write_image.SetSpacing(desired_spacing)
    to_write_image.SetDirection(image.GetDirection())

    # Predict the limited FOV segmentation
    limited_fov_segmentation_array = predict.predict_from_array_by_iterator(limited_fov_image_array, target_model,
                                                                            accelerator, nnunet_log_filename)

    # Expand the segmentation to the original FOV
    expanded_segmentation_array = expand_segmentation_fov(limited_fov_segmentation_array, original_fov_info)

    # Limit the FOV again based on label intensities and largest component condition
    limited_fov_segmentation_array, original_fov_info = limit_fov(expanded_segmentation_array, resampled_to_crop_segmentation_array,
                                                                  target_model_fov_information["label_intensity_to_crop_from"],
                                                                  target_model_fov_information["largest_component_only"])

    # Expand the segmentation array to the original FOV
    segmentation_array = expand_segmentation_fov(limited_fov_segmentation_array, original_fov_info)

    # Return the final model and segmentation array
    return target_model, segmentation_array, desired_spacing


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

        if CHUNK_THRESHOLD <= 0:
            raise ValueError('CHUNK_THRESHOLD must be greater than 0')

        # If the axis is smaller than the threshold, it cannot be split into smaller chunks
        if axis < CHUNK_THRESHOLD:
            return 1

        # Determine the maximum number of chunks that the axis can be split into
        split = axis // CHUNK_THRESHOLD

        # Reduce the number of chunks until the axis is evenly divisible by split
        while axis % split != 0:
            split -= 1

        return split

    @staticmethod
    def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: tuple, interpolation_method: int,
                                 output_spacing: tuple, output_size: tuple) -> da.array:
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
                                      output_spacing: tuple = (1.5, 1.5, 1.5),
                                      output_size: tuple = None) -> SimpleITK.Image:
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
                         output_image_path: str = None, is_label_image: bool = False) -> SimpleITK.Image:
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
                                            output_spacing: tuple = (1.5, 1.5, 1.5),
                                            output_size: tuple = None) -> np.array:
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
