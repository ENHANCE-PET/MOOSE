"""Exercise real imaging operations with generated voxel arrays only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import SimpleITK as sitk

from moosez import image_processing


def test_synthetic_nifti_orientation_resampling_and_chunk_roundtrip(tmp_path):
    array = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 2.0, 2.0))
    source = tmp_path / 'CT_synthetic.nii.gz'
    sitk.WriteImage(image, str(source))
    loaded = image_processing.image_read(str(source))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded), array)
    oriented = image_processing.image_reorient(loaded, 'RAS')
    assert image_processing.image_get_orientation_code(oriented) == 'RAS'
    sampled = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(
        loaded, 'linear', (1.0, 1.0, 1.0))
    assert sampled.shape == (32, 32, 32)
    assert np.isfinite(sampled).all()
    chunks, positions = image_processing.ImageChunker.array_to_chunks(
        array[None], (1, 2, 2, 2), (0, 2, 2, 2))
    reconstructed = image_processing.ImageChunker.chunks_to_array(chunks, positions, array[None].shape)
    np.testing.assert_array_equal(reconstructed, array[None])


def test_synthetic_mask_statistics_are_written_with_expected_volume_and_mean(tmp_path):
    image = sitk.GetImageFromArray(np.arange(512, dtype=np.float32).reshape(8, 8, 8))
    image.SetSpacing((2.0, 2.0, 2.0))
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    mask = sitk.GetImageFromArray(labels)
    mask.CopyInformation(image)
    model_metadata = SimpleNamespace(organ_indices={1: 'synthetic region'})
    volume_csv = tmp_path / 'volumes.csv'
    intensity_csv = tmp_path / 'intensities.csv'
    image_processing.get_shape_statistics(mask, model_metadata, str(volume_csv))
    image_processing.get_intensity_statistics(image, mask, model_metadata, str(intensity_csv))
    volumes = pd.read_csv(volume_csv, index_col=0)
    intensities = pd.read_csv(intensity_csv, index_col=0)
    assert volumes.loc[1, 'Volume(mm3)'] == 512.0
    assert volumes.loc[1, 'Regions-Present'] == 'synthetic region'
    assert intensities.loc[1, 'Mean'] == 255.5
