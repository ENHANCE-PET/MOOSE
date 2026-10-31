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


def test_synthetic_dicom_series_converts_with_pydicom3_without_native_fallback(tmp_path, monkeypatch):
    """The declared converter must work with pydicom 3, which removed read_file."""
    import nibabel as nib
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    from moosez import image_conversion

    source = tmp_path / 'synthetic_ct'
    source.mkdir()
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for index in range(8):
        instance_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = instance_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(source / f'slice_{index}.dcm'), {}, file_meta=meta, preamble=b'\0' * 128)
        dataset.SOPClassUID, dataset.SOPInstanceUID = CTImageStorage, instance_uid
        dataset.StudyInstanceUID, dataset.SeriesInstanceUID = study_uid, series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.PatientName, dataset.PatientID = 'Synthetic^Phantom', 'SYNTHETIC'
        dataset.Modality, dataset.Manufacturer = 'CT', 'Synthetic'
        dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(index * 2)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing, dataset.SliceThickness = [1.0, 1.0], 2.0
        dataset.Rows, dataset.Columns = 8, 8
        dataset.SamplesPerPixel, dataset.PhotometricInterpretation = 1, 'MONOCHROME2'
        dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 16, 16, 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope, dataset.RescaleIntercept = 1, 0
        dataset.PixelData = np.full((8, 8), index * 10, dtype='<i2').tobytes()
        dataset.save_as(dataset.filename, enforce_file_format=True)

    def unavailable_native_converter(*args, **kwargs):
        raise OSError('Native fallback is deliberately unavailable for this Python-converter regression')

    monkeypatch.setattr(image_conversion.subprocess, 'run', unavailable_native_converter)
    output = image_conversion.dicom_to_nifti(str(source), str(tmp_path), compression=True)
    assert output is not None, 'Python DICOM conversion must succeed without the native fallback'
    converted = nib.load(output)
    assert converted.shape == (8, 8, 8)
    assert converted.header.get_zooms() == (1.0, 1.0, 2.0)
    values, counts = np.unique(converted.get_fdata(), return_counts=True)
    np.testing.assert_array_equal(values, [0, 10, 20, 30, 40, 50, 60, 70])
    np.testing.assert_array_equal(counts, [64] * 8)
