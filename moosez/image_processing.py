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
import numpy as np
import pandas as pd
import nibabel
import os
import tempfile
from typing import Dict
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
    # voxels per label times the volume of a voxel (what LabelShapeStatisticsImageFilter's GetPhysicalSize gives)
    voxel_counts = label_voxel_counts(mask_image)
    voxel_volume = float(np.prod(mask_image.GetSpacing()))
    columns = ['Volume(mm3)']
    stats_df = pd.DataFrame(data=[(count * voxel_volume,) for count in voxel_counts.values()],
                            index=list(voxel_counts), columns=columns)

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


def label_voxel_counts(label_image: SimpleITK.Image) -> Dict[int, int]:
    """
    The number of voxels of every label in the image, without the background, in ascending order of the labels.

    :param label_image: The multilabel mask image.
    :type label_image: sitk.Image
    :return: The voxel count per label present.
    :rtype: Dict[int, int]
    """
    labels = SimpleITK.GetArrayViewFromImage(label_image)
    counts = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    for label_slice in labels:  # per slice: np.bincount copies what it counts to int64
        counts += np.bincount(label_slice.ravel(), minlength=len(counts))
    return {int(label): int(counts[label]) for label in np.flatnonzero(counts) if label != 0}


def image_get_orientation_code(image: SimpleITK.Image) -> str:
    image_orientation_code = SimpleITK.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    return image_orientation_code


def image_read(image_path: str) -> SimpleITK.Image:
    SimpleITK.ProcessObject_SetGlobalWarningDisplay(False)
    try:
        image = SimpleITK.ReadImage(image_path)
    except RuntimeError:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_tmp_path = os.path.join(tmpdir, "non_orthonormal.nii.gz")
            image_nibabel = nibabel.load(image_path)
            nibabel.save(nibabel.Nifti1Image(image_nibabel.get_fdata(), image_nibabel.get_qform()), file_tmp_path)
            image = SimpleITK.ReadImage(file_tmp_path)
    SimpleITK.ProcessObject_SetGlobalWarningDisplay(True)
    return image


def reslice_label_image(reference_image: SimpleITK.Image, label_image: SimpleITK.Image) -> SimpleITK.Image:
    """
    Reslices a label image onto the grid of another image (nearest neighbour), as int32.

    :param reference_image: The image whose grid the labels go to.
    :type reference_image: SimpleITK.Image
    :param label_image: The label image to reslice.
    :type label_image: SimpleITK.Image
    :return: The resliced label image.
    :rtype: SimpleITK.Image
    """
    resliced = SimpleITK.Resample(label_image, reference_image, SimpleITK.Transform(), SimpleITK.sitkNearestNeighbor)
    return SimpleITK.Cast(resliced, SimpleITK.sitkInt32)
