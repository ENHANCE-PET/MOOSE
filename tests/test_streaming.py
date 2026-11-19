"""The pipeline (moosez/streaming.py) with a tiny randomly initialised nnU-Net model, on CPU."""
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK
import torch
from scipy import ndimage

from moosez import constants, file_utilities, image_processing, models, streaming, system

LAS = SimpleITK.DICOMOrientImageFilter_GetDirectionCosinesFromOrientation("LAS")
TINY_MODEL = "clin_ct_tiny"
TINY_FOLDER = "Dataset999_Tiny"
TINY_LABELS = ["liver", "spleen", "kidney_left", "kidney_right"]


def _tiny_plans(reader: str) -> dict:
    return {
        "dataset_name": TINY_FOLDER, "plans_name": "nnUNetPlans", "image_reader_writer": reader,
        "original_median_spacing_after_transp": [2.0, 2.0, 2.0], "original_median_shape_after_transp": [32, 32, 32],
        "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2], "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {"0": {"max": 1000., "mean": 0., "median": 0., "min": -1000.,
                                                              "percentile_00_5": -900., "percentile_99_5": 900., "std": 300.}},
        "configurations": {"3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres", "preprocessor_name": "DefaultPreprocessor", "batch_size": 2,
            "patch_size": [32, 32, 32], "median_image_size_in_voxels": [32, 32, 32], "spacing": [2.0, 2.0, 2.0],
            "normalization_schemes": ["CTNormalization"], "use_mask_for_norm": [False],
            "resampling_fn_data": "resample_torch_fornnunet", "resampling_fn_seg": "resample_torch_fornnunet",
            "resampling_fn_probabilities": "resample_torch_fornnunet",
            "resampling_fn_data_kwargs": {"is_seg": False, "force_separate_z": False, "memefficient_seg_resampling": True},
            "resampling_fn_seg_kwargs": {"is_seg": True, "force_separate_z": False, "memefficient_seg_resampling": True},
            "resampling_fn_probabilities_kwargs": {"is_seg": False, "force_separate_z": False, "memefficient_seg_resampling": True},
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {"n_stages": 3, "features_per_stage": [4, 8, 16], "conv_op": "torch.nn.modules.conv.Conv3d",
                                "kernel_sizes": [[3, 3, 3]] * 3, "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                                "n_conv_per_stage": [1, 1, 1], "n_conv_per_stage_decoder": [1, 1], "conv_bias": True,
                                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                                "norm_op_kwargs": {"eps": 1e-05, "affine": True}, "dropout_op": None,
                                "dropout_op_kwargs": None, "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True}},
                "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]},
            "batch_dice": True}},
    }


def make_tiny_model(models_directory: str, reader: str = "SimpleITKIOWithReorient", folder: str = TINY_FOLDER) -> str:
    """A randomly initialised nnU-Net model folder in moosez's layout; returns the configuration folder."""
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    plans = _tiny_plans(reader)
    dataset = {"channel_names": {"0": "CT"}, "labels": {"background": 0, **{n: i + 1 for i, n in enumerate(TINY_LABELS)}},
               "numTraining": 1, "file_ending": ".nii.gz"}
    configuration_directory = os.path.join(models_directory, folder, "nnUNetTrainer__nnUNetPlans__3d_fullres")
    os.makedirs(os.path.join(configuration_directory, "fold_0"))
    with open(os.path.join(configuration_directory, "plans.json"), "w") as f:
        json.dump(plans, f)
    with open(os.path.join(configuration_directory, "dataset.json"), "w") as f:
        json.dump(dataset, f)
    architecture = plans["configurations"]["3d_fullres"]["architecture"]
    torch.manual_seed(0)
    network = get_network_from_plans(architecture["network_class_name"], architecture["arch_kwargs"],
                                     architecture["_kw_requires_import"], 1, len(TINY_LABELS) + 1, allow_init=True,
                                     deep_supervision=False)
    torch.save({"network_weights": network.state_dict(), "trainer_name": "nnUNetTrainer",
                "init_args": {"configuration": "3d_fullres"}, "inference_allowed_mirroring_axes": None},
               os.path.join(configuration_directory, "fold_0", "checkpoint_final.pth"))
    return configuration_directory


def register(monkeypatch, tmp_path, identifier: str = TINY_MODEL, plans_reader: str = "SimpleITKIOWithReorient", **fields):
    """The tiny model in a temporary models directory (its plans name plans_reader) and its registry entry (with fields)."""
    models_directory = str(tmp_path / "models")
    if not os.path.isdir(models_directory):
        make_tiny_model(models_directory, plans_reader)
    monkeypatch.setattr(system, "MODELS_DIRECTORY_PATH", models_directory)
    monkeypatch.setitem(models.MODEL_METADATA, identifier, {
        constants.KEY_URL: f"{constants.LOCAL_URL_PREFIX}tiny", constants.KEY_FOLDER_NAME: TINY_FOLDER, **fields})
    return identifier


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    return register(monkeypatch, tmp_path)


def register_with_roi(monkeypatch, tmp_path) -> tuple:
    """Two tiny models: one whose label map gives the roi (labels 2, 3) of the other, which keeps the slices of 3."""
    roi_model = register(monkeypatch, tmp_path, "clin_ct_tinyroi")
    with_roi = register(monkeypatch, tmp_path, **{constants.KEY_ROI: {"model": roi_model, "crop_labels": (2, 3), "keep_label": 3}})
    return roi_model, with_roi


def same_grid(image: SimpleITK.Image, other: SimpleITK.Image) -> bool:
    return (image.GetSize() == other.GetSize() and np.allclose(image.GetSpacing(), other.GetSpacing(), atol=1e-4)
            and np.allclose(image.GetOrigin(), other.GetOrigin(), atol=1e-4)
            and np.allclose(image.GetDirection(), other.GetDirection(), atol=1e-4))


def predict(model_identifier: str, images: list) -> list:
    with streaming.MoosePredictor(model_identifier, "cpu") as predictor:
        return [result[model_identifier] for result in predictor.predict(images)]


def synthetic_ct(seed: int = 0, shape=(30, 36, 40)) -> SimpleITK.Image:
    """A smooth random volume (z, y, x) stored LAS, like the ENHANCE.PET CTs."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    array = 800 * np.sin(3 * x + rng.uniform(0, 3)) * np.cos(2 * y + rng.uniform(0, 3)) + 300 * z
    array += rng.normal(0, 50, shape)
    image = SimpleITK.GetImageFromArray(array.astype(np.int16))
    image.SetSpacing((1.5, 1.4, 2.5))
    image.SetOrigin((10., -20., 30.))
    image.SetDirection(LAS)
    return image


def test_image_to_case_matches_the_readers(tmp_path):
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO, SimpleITKIOWithReorient

    las = synthetic_ct()
    las_file = str(tmp_path / "las.nii.gz")
    SimpleITK.WriteImage(las, las_file)
    for reader_name, reader in (("SimpleITKIO", SimpleITKIO), ("SimpleITKIOWithReorient", SimpleITKIOWithReorient)):
        expected_array, expected_properties = reader().read_images([las_file])
        array, properties = streaming.image_to_case(SimpleITK.ReadImage(las_file), reader_name)
        np.testing.assert_array_equal(array, expected_array)
        assert array.dtype == expected_array.dtype
        assert properties["spacing"] == expected_properties["spacing"]
        for key, value in expected_properties["sitk_stuff"].items():
            assert properties["sitk_stuff"][key] == value, key

    # the reorienting reader gives every storage orientation the same (RAS) arrays
    ras_array, _ = SimpleITKIOWithReorient().read_images([las_file])
    for code in ("RAS", "LPS", "RPI"):
        array, _ = streaming.image_to_case(SimpleITK.DICOMOrient(las, code), "SimpleITKIOWithReorient")
        np.testing.assert_array_equal(array, ras_array)


@pytest.mark.parametrize("reader", ["SimpleITKIO", "SimpleITKIOWithReorient"])
def test_case_to_image_restores_the_input_geometry(reader):
    for code in ("LAS", "RAS", "LPS"):
        image = SimpleITK.DICOMOrient(synthetic_ct(), code)
        array, properties = streaming.image_to_case(image, reader)
        labels = (np.abs(array[0]).astype(np.int64) % 7).astype(np.uint8)   # any label map in the case's arrays
        restored = streaming.case_to_image(labels, properties, reader)
        assert restored.GetSize() == image.GetSize()
        np.testing.assert_allclose(restored.GetDirection(), image.GetDirection())
        np.testing.assert_allclose(restored.GetOrigin(), image.GetOrigin(), atol=1e-9)
        np.testing.assert_allclose(restored.GetSpacing(), image.GetSpacing())
        expected = (np.abs(SimpleITK.GetArrayFromImage(image)).astype(np.int64) % 7).astype(np.uint8)
        np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(restored), expected)


def test_prediction_does_not_depend_on_the_storage_orientation(tiny_model):
    codes = ("LAS", "RAS", "LPS", "RPI")
    outputs = dict(zip(codes, predict(tiny_model, [SimpleITK.DICOMOrient(synthetic_ct(), code) for code in codes])))
    reference = SimpleITK.GetArrayFromImage(outputs["LAS"])
    assert len(np.unique(reference)) > 1, "the tiny model predicts a single label; the test would be vacuous"
    for code, output in outputs.items():
        assert image_processing.image_get_orientation_code(output) == code
        np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(SimpleITK.DICOMOrient(output, "LAS")), reference)


def test_the_registry_reader_overrides_the_plans(tmp_path, monkeypatch):
    """Plans that name SimpleITKIO (the models trained on images moosez had reoriented): the registry's reader makes
    the prediction independent of the storage orientation; without it the model gets the arrays as stored."""
    as_planned = register(monkeypatch, tmp_path, "clin_ct_tinyplans", plans_reader="SimpleITKIO")
    overridden = register(monkeypatch, tmp_path, plans_reader="SimpleITKIO", **{constants.KEY_READER: "SimpleITKIOWithReorient"})
    ras = SimpleITK.DICOMOrient(synthetic_ct(), "RAS")
    images = [ras, SimpleITK.DICOMOrient(ras, "LAS")]
    reference, from_las = [SimpleITK.GetArrayFromImage(SimpleITK.DICOMOrient(o, "RAS")) for o in predict(overridden, images)]
    np.testing.assert_array_equal(from_las, reference)
    stored_ras, stored_las = [SimpleITK.GetArrayFromImage(SimpleITK.DICOMOrient(o, "RAS")) for o in predict(as_planned, images)]
    np.testing.assert_array_equal(stored_ras, reference)                # a RAS image is the same for both readers
    assert (stored_las != reference).mean() > 0.01                      # the mirrored array gives another label map


def test_the_registry_orientation_is_the_one_the_model_gets(tmp_path, monkeypatch):
    """KEY_ORIENTATION: a model trained on LAS arrays gets every image as its LAS array and gives the label map back
    in the image's own orientation."""
    las_model = register(monkeypatch, tmp_path, plans_reader="SimpleITKIO",
                         **{constants.KEY_READER: "SimpleITKIOWithReorient", constants.KEY_ORIENTATION: "LAS"})
    as_stored = register(monkeypatch, tmp_path, "clin_ct_tinyplans", plans_reader="SimpleITKIO")
    las = synthetic_ct()
    array, _ = streaming.image_to_case(SimpleITK.DICOMOrient(las, "RPI"), "SimpleITKIOWithReorient", "LAS")
    np.testing.assert_array_equal(array[0], SimpleITK.GetArrayFromImage(las))
    reference = SimpleITK.GetArrayFromImage(predict(as_stored, [las])[0])            # the LAS array as it is stored
    for code in ("LAS", "RAS", "LPS"):
        output = predict(las_model, [SimpleITK.DICOMOrient(las, code)])[0]
        assert image_processing.image_get_orientation_code(output) == code
        np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(SimpleITK.DICOMOrient(output, "LAS")), reference)


def test_several_subjects_equal_single_ones(tiny_model, tmp_path):
    files = []
    for seed in range(3):
        files.append(str(tmp_path / f"ct_{seed}.nii.gz"))
        SimpleITK.WriteImage(SimpleITK.DICOMOrient(synthetic_ct(seed), ("LAS", "RAS", "LPS")[seed]), files[-1])
    together = predict(tiny_model, files)
    for file, image in zip(files, together):
        single = predict(tiny_model, [file])[0]
        np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(image), SimpleITK.GetArrayFromImage(single))
        assert image.GetDirection() == single.GetDirection()


def test_the_first_image_is_read_while_the_models_load(tiny_model, monkeypatch):
    """predict() starts to read before the networks are built, so reading and loading overlap: here the build waits
    until a reader thread has the image's array, which never happens if the models are built first."""
    read, built_after_read = threading.Event(), []
    image_to_case = streaming.image_to_case
    initialize = streaming.nnUNetPredictorStreaming.initialize_from_folder_base_trainer

    def reading(*args, **kwargs):
        read.set()
        return image_to_case(*args, **kwargs)

    def building(self, *args, **kwargs):
        built_after_read.append(read.wait(60))
        return initialize(self, *args, **kwargs)

    monkeypatch.setattr(streaming, "image_to_case", reading)
    monkeypatch.setattr(streaming.nnUNetPredictorStreaming, "initialize_from_folder_base_trainer", building)
    image = synthetic_ct()
    with streaming.MoosePredictor(tiny_model, "cpu") as predictor:
        assert predictor._predictors is None                 # nothing is built before the first use
        result = predictor.predict([image])[0][tiny_model]
    assert built_after_read == [True] and same_grid(result, image)


@pytest.mark.parametrize("shape, new_shape", [((40, 37, 23), (21, 37, 52)), ((12, 30, 30), (40, 13, 14))])
def test_resample_cubic_is_scipy_zoom_with_aligned_voxel_centres(shape, new_shape):
    data = np.random.default_rng(0).normal(size=(2, *shape)).astype(np.float32)
    expected = np.stack([ndimage.zoom(channel.astype(np.float64), np.array(new_shape) / np.array(shape), order=3,
                                      mode='nearest', grid_mode=True) for channel in data])
    resampled = streaming.resample_cubic(data, new_shape)
    assert resampled.shape == (2, *new_shape) and resampled.dtype == np.float32
    np.testing.assert_allclose(resampled, expected, atol=1e-4)
    np.testing.assert_array_equal(streaming.resample_cubic(data, shape), data)


def test_cubic_input_resampling_reaches_the_preprocessing(tmp_path, monkeypatch):
    """A model with KEY_INPUT_RESAMPLING: nnU-Net's preprocessing (reorient, normalise) with resample_cubic in place of
    the plans' function, and the label map still comes out on the input grid."""
    planned = register(monkeypatch, tmp_path, "clin_ct_tinyplans")
    cubic = register(monkeypatch, tmp_path, **{constants.KEY_INPUT_RESAMPLING: constants.INPUT_RESAMPLING_CUBIC})
    image = synthetic_ct()
    with streaming.MoosePredictor([planned, cubic], "cpu") as predictor:
        assert predictor.predictors[cubic].force_trilinear and not predictor.predictors[planned].force_trilinear
        array, properties = streaming.image_to_case(image, "SimpleITKIOWithReorient")
        linear, _ = predictor.preprocess(predictor.models[planned], array, properties)
        data, _ = predictor.preprocess(predictor.models[cubic], array, properties)
        (result,) = predictor.predict([image])
    intensities = models.Model(cubic, system.OutputManager(False, False)).plans["foreground_intensity_properties_per_channel"]["0"]
    normalised = (np.clip(array[0].astype(np.float32), intensities["percentile_00_5"], intensities["percentile_99_5"])
                  - intensities["mean"]) / intensities["std"]
    expected = ndimage.zoom(normalised.astype(np.float64), np.array(data.shape[1:]) / np.array(array.shape[1:]), order=3,
                            mode='nearest', grid_mode=True)
    assert data.shape == linear.shape != array.shape
    np.testing.assert_allclose(data[0], expected, atol=1e-4)
    assert np.abs(data - linear).max() > 1e-3
    assert same_grid(result[cubic], image) and same_grid(result[planned], image)


# the reader the plans of a released model name (the others: SimpleITKIO)
RELEASED_PLANS_READER = {"clin_ct_anatomy": "NibabelIOWithReorient"}


@pytest.mark.parametrize("identifier", list(models.MODEL_METADATA))
def test_every_registry_entry_runs_through_the_pipeline(identifier, tmp_path, monkeypatch):
    """Each entry of the real registry with a tiny model in place of its weights (plans with the reader the released
    model's plans name): the pipeline resolves its name, reader, input resampling and roi model, and predicts."""
    models_directory = str(tmp_path / "models")
    monkeypatch.setattr(system, "MODELS_DIRECTORY_PATH", models_directory)
    entry, needed = models.MODEL_METADATA[identifier], [identifier]
    while constants.KEY_ROI in models.MODEL_METADATA[needed[-1]]:
        needed.append(models.MODEL_METADATA[needed[-1]][constants.KEY_ROI]["model"])
    for name in needed:
        folder = models.MODEL_METADATA[name][constants.KEY_FOLDER_NAME]
        make_tiny_model(models_directory, RELEASED_PLANS_READER.get(name, "SimpleITKIO"), folder)
        with open(os.path.join(models_directory, folder, "model_version.json"), "w") as f:     # so nothing is downloaded
            json.dump({"url": models.MODEL_METADATA[name][constants.KEY_URL]}, f)
    cubic = entry.get(constants.KEY_INPUT_RESAMPLING) == constants.INPUT_RESAMPLING_CUBIC
    assert entry.get(constants.KEY_INPUT_RESAMPLING) in (None, constants.INPUT_RESAMPLING_CUBIC)
    image = synthetic_ct()
    with streaming.MoosePredictor(identifier, "cpu") as predictor:
        model, nnunet_predictor = predictor.requested[0], predictor.predictors[identifier]
        assert set(predictor.models) == set(needed)
        plans_reader = RELEASED_PLANS_READER.get(identifier, "SimpleITKIO")
        assert model.reader == entry.get(constants.KEY_READER, constants.SIMPLEITK_READER_FOR.get(plans_reader, plans_reader))
        assert model.reader in constants.SUPPORTED_READERS
        assert model.orientation == entry.get(constants.KEY_ORIENTATION, "RAS") and len(model.orientation) == 3
        assert isinstance(nnunet_predictor.configuration_manager, streaming.CubicInput) == cubic
        assert nnunet_predictor.force_trilinear == cubic
        if model.roi is not None:
            assert model.roi["keep_label"] in model.roi["crop_labels"]
        (result,) = predictor.predict([image])
    # the tiny roi model has none of the roi's labels, so a model with a roi gives no label map here
    assert result[identifier] is None if model.roi is not None else same_grid(result[identifier], image)


def test_label_slab():
    labels = np.zeros((20, 6, 6), dtype=np.uint8)
    labels[3:5, 1, 1] = 22          # a small component of the label
    labels[8:12, 2:5, 2:5] = 22     # its largest component
    labels[6:8] = 21
    labels[14, 0, 0] = 24
    assert streaming.label_slab(labels, (20, 21, 22, 23, 24)) == (3, 15)
    assert streaming.label_slab(labels, [22]) == (3, 12)
    assert streaming.label_slab(labels, [22], largest_component=True) == (8, 12)
    assert streaming.label_slab(labels, [20]) is None and streaming.label_slab(labels, [300]) is None


def test_a_model_with_a_roi_runs_on_the_slab_and_keeps_the_slices_of_its_label(tmp_path, monkeypatch):
    """The roi step: the model sees only the slices with crop_labels of the roi model's label map; its label map is
    pasted into the full grid, restricted to the slices of keep_label. Without the structures: no result."""
    roi_model, with_roi = register_with_roi(monkeypatch, tmp_path)
    image = SimpleITK.DICOMOrient(synthetic_ct(), "RAS")
    roi_labels = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    roi_labels[6:9] = 2
    roi_labels[12:18, 5:20, 5:20] = 3
    roi_labels[22, 0, 0] = 3                                      # a stray voxel: widens the slab, not the kept slices
    with streaming.MoosePredictor(with_roi, "cpu") as predictor:
        assert set(predictor.models) == {with_roi, roi_model}
        case = streaming.Case(image)
        case.segmentations[roi_model] = roi_labels                # in place of the roi model's label map
        segmentation = predictor.segment(predictor.models[with_roi], case)
        empty = streaming.Case(image)
        empty.segmentations[roi_model] = np.zeros_like(roi_labels)
        assert predictor.segment(predictor.models[with_roi], empty) is None
    expected = SimpleITK.GetArrayFromImage(predict(roi_model, [image[:, :, 6:23]])[0])    # same weights, no roi
    assert segmentation.shape == roi_labels.shape and len(np.unique(expected[6:12])) > 1
    np.testing.assert_array_equal(segmentation[12:18], expected[12 - 6:18 - 6])
    assert not segmentation[:12].any() and not segmentation[18:].any()


def test_run_files_writes_the_outputs(tiny_model, tmp_path):
    input_directory, output_directory = tmp_path / "in", tmp_path / "out"
    input_directory.mkdir()
    for seed, code in enumerate(("LAS", "RAS")):
        SimpleITK.WriteImage(SimpleITK.DICOMOrient(synthetic_ct(seed), code), str(input_directory / f"case{seed}.nii.gz"))
    files = streaming.collect_image_files([str(input_directory)])
    information = streaming.run_files(files, str(output_directory), [tiny_model], "cpu", system.OutputManager(False, False),
                                      single_labels=True, statistics=True)
    prefix = "clin_CT_tiny_"
    for seed in range(2):
        segmentation = output_directory / f"{prefix}segmentation_case{seed}.nii.gz"
        assert segmentation.is_file()
        assert (output_directory / "stats" / f"{prefix}case{seed}_volume.csv").is_file()
        assert (output_directory / "stats" / f"{prefix}case{seed}_CT_intensity.csv").is_file()
        source = SimpleITK.ReadImage(str(input_directory / f"case{seed}.nii.gz"))
        assert same_grid(SimpleITK.ReadImage(str(segmentation)), source)
        single_labels = list((output_directory / constants.SINGLE_LABELS_FOLDER / f"{prefix}case{seed}").iterdir())
        assert single_labels and all(f.name[:-len(".nii.gz")] in TINY_LABELS for f in single_labels)
    organ_indices = json.load(open(output_directory / f"{prefix}organ_indices.json"))["organ_indices"]
    assert [v["name"] for v in organ_indices.values()] == TINY_LABELS
    assert set(information[tiny_model]["cases"]) == {"case0", "case1"}


def test_a_broken_image_does_not_stop_the_batch(tiny_model, tmp_path):
    input_directory, output_directory = tmp_path / "in", tmp_path / "out"
    input_directory.mkdir()
    for seed in range(2):
        SimpleITK.WriteImage(synthetic_ct(seed), str(input_directory / f"case{seed}.nii.gz"))
    (input_directory / "broken.nii.gz").write_bytes(b"not an image")
    files = streaming.collect_image_files([str(input_directory)])
    information = streaming.run_files(files, str(output_directory), [tiny_model], "cpu", system.OutputManager(False, False),
                                      statistics=True)
    assert set(information[tiny_model]["cases"]) == {"case0", "case1"}
    assert not (output_directory / "clin_CT_tiny_segmentation_broken.nii.gz").exists()
    for seed in range(2):
        assert (output_directory / "stats" / f"clin_CT_tiny_case{seed}_volume.csv").is_file()


def test_moose_api_with_a_streaming_model(tiny_model, tmp_path):
    import moosez
    image = SimpleITK.DICOMOrient(synthetic_ct(1), "RAS")
    file = str(tmp_path / "CT_case.nii.gz")
    SimpleITK.WriteImage(image, file)
    (written,), (model,) = moosez.moose(file, tiny_model, str(tmp_path), "cpu")
    assert written == str(tmp_path / "clin_CT_tiny_segmentation_CT_case.nii.gz") and os.path.isfile(written)
    (from_image,), _ = moosez.moose(image, tiny_model, accelerator="cpu")
    assert same_grid(from_image, image)
    np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(from_image), SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(written)))
    array = SimpleITK.GetArrayFromImage(synthetic_ct(1))
    (from_array,), _ = moosez.moose((array, (1.5, 1.4, 2.5)), tiny_model, accelerator="cpu")
    assert from_array.shape == array.shape and model.model_identifier == tiny_model


def test_statistics_of_an_image_on_another_grid(tmp_path):
    """PET statistics of a CT label map: the label map is resliced onto the PET grid with nearest neighbour."""
    rng = np.random.default_rng(3)
    labels = SimpleITK.GetImageFromArray(rng.integers(0, 4, (30, 36, 40)).astype(np.uint8))
    labels.SetSpacing((1.0, 1.0, 2.0))
    pet = SimpleITK.GetImageFromArray(rng.gamma(2.0, 3000.0, (20, 18, 22)).astype(np.float32))
    pet.SetSpacing((2.0, 2.2, 3.0))
    pet.SetOrigin((3.0, -1.0, 2.0))
    model = SimpleNamespace(organ_indices={1: "a", 2: "b", 3: "c"}, modality_full="CT")

    resliced = image_processing.reslice_label_image(pet, labels)
    image_processing.get_intensity_statistics(pet, resliced, model, str(tmp_path / "expected_pt.csv"))
    image_processing.get_shape_statistics(labels, model, str(tmp_path / "expected_volume.csv"))
    streaming.write_statistics(labels, model, str(tmp_path / "volume.csv"), {"PT": pet}, {"PT": str(tmp_path / "pt.csv")})
    for ours, expected in (("pt.csv", "expected_pt.csv"), ("volume.csv", "expected_volume.csv")):
        assert (tmp_path / ours).read_text() == (tmp_path / expected).read_text()


def test_export_statistics_use_the_label_map_and_image_in_memory(tmp_path):
    """export_case's statistics = moosez's statistics functions on the label map it returns and the input image."""
    image = SimpleITK.DICOMOrient(synthetic_ct(2), "RAS")
    array, properties = streaming.image_to_case(image, "SimpleITKIOWithReorient")
    labels = (np.abs(array[0]).astype(np.int64) % 4).astype(np.uint8)
    model = SimpleNamespace(organ_indices={1: "a", 2: "b", 3: "c"}, reader="SimpleITKIOWithReorient",
                            modality_full="CT")
    output = {"statistics": {"volume_csv": str(tmp_path / "stats" / "volume.csv"),
                             "intensity": {"CT": str(tmp_path / "stats" / "ct.csv")}}}
    label_image = streaming.export_case(labels, properties, model, output, {"CT": image})
    assert streaming.export_case(labels, properties, model, {"file": str(tmp_path / "seg.nii.gz")}) == \
        str(tmp_path / "seg.nii.gz")                                                # a written label map: its path
    np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(tmp_path / "seg.nii.gz"))),
                                  SimpleITK.GetArrayFromImage(label_image))
    image_processing.get_shape_statistics(label_image, model, str(tmp_path / "expected_volume.csv"))
    image_processing.get_intensity_statistics(image, label_image, model, str(tmp_path / "expected_ct.csv"))
    assert (tmp_path / "stats" / "volume.csv").read_text() == (tmp_path / "expected_volume.csv").read_text()
    assert (tmp_path / "stats" / "ct.csv").read_text() == (tmp_path / "expected_ct.csv").read_text()


def test_a_subject_keeps_only_the_voxels_the_call_still_needs(tiny_model, tmp_path, monkeypatch):
    """After the reader thread, the image and the reader arrays are dropped unless the intensity statistics or a model
    with a roi still read them; the properties stay for the export."""
    image = SimpleITK.DICOMOrient(synthetic_ct(), "RAS")
    with streaming.MoosePredictor(tiny_model, "cpu") as predictor:
        modality = predictor.models[tiny_model].modality_full
        case = predictor._read({modality: image}, False)[modality]
        assert case.image is None and case.arrays
        assert all(array is None and properties for array, properties in case.arrays.values())
        assert predictor._read({modality: image}, True)[modality].image is image
    roi_model, with_roi = register_with_roi(monkeypatch, tmp_path)
    with streaming.MoosePredictor(with_roi, "cpu") as predictor:
        modality = predictor.models[with_roi].modality_full
        case = predictor._read({modality: image}, False)[modality]
        assert case.image is None and case.array(predictor.models[with_roi])[0] is not None
        assert predictor._roi_models == {roi_model}     # the label map of this model is kept for the one that crops


def test_statistics_of_a_second_image_whose_grid_is_almost_the_label_maps(tmp_path):
    """A PET resampled to the CT grid differs from it by rounding: more than SimpleITK's statistics filter accepts,
    less than a grid comparison would notice. The label map is resliced onto the image of another modality always."""
    rng = np.random.default_rng(3)
    labels = SimpleITK.GetImageFromArray(rng.integers(0, 4, (12, 14, 16)).astype(np.uint8))
    labels.SetOrigin((-250.0, -250.0, 100.0))
    pet = SimpleITK.GetImageFromArray(rng.gamma(2.0, 3000.0, (12, 14, 16)).astype(np.float32))
    pet.SetOrigin((-250.0 + 5e-5, -250.0, 100.0))
    model = SimpleNamespace(organ_indices={1: "a", 2: "b", 3: "c"}, modality_full="CT")
    with pytest.raises(RuntimeError):                               # what the filter does with these two as they are
        image_processing.get_intensity_statistics(pet, labels, model, str(tmp_path / "direct.csv"))
    streaming.write_statistics(labels, model, str(tmp_path / "volume.csv"), {"PT": pet}, {"PT": str(tmp_path / "pt.csv")})
    assert len((tmp_path / "pt.csv").read_text().splitlines()) == 4


def test_volume_statistics_and_single_label_files(tmp_path):
    """Volumes are voxel counts times the voxel volume, as SimpleITK's shape statistics give them; -sl writes one
    binary file per label present, on the label map's grid."""
    rng = np.random.default_rng(5)
    array = rng.integers(0, 6, (9, 10, 11)).astype(np.uint8)
    array[array == 4] = 0                                           # label 4 is not present
    labels = SimpleITK.GetImageFromArray(array)
    labels.SetSpacing((0.7, 1.3, 2.9))
    labels.SetDirection(LAS)
    names = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
    model = SimpleNamespace(organ_indices=names)
    assert image_processing.label_voxel_counts(labels) == {l: int((array == l).sum()) for l in (1, 2, 3, 5)}
    image_processing.get_shape_statistics(labels, model, str(tmp_path / "volume.csv"))
    shape_filter = SimpleITK.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(labels)
    lines = (tmp_path / "volume.csv").read_text().splitlines()
    assert lines[0] == ",Regions-Present,Volume(mm3)"
    assert [line.split(",") for line in lines[1:]] == \
        [[str(l), names[l], str(shape_filter.GetPhysicalSize(l))] for l in (1, 2, 3, 5)]

    streaming.write_single_labels(labels, names, str(tmp_path / "single"))
    assert sorted(f.name for f in (tmp_path / "single").iterdir()) == [f"{names[l]}.nii.gz" for l in (1, 2, 3, 5)]
    single = SimpleITK.ReadImage(str(tmp_path / "single" / "c.nii.gz"))
    np.testing.assert_array_equal(SimpleITK.GetArrayFromImage(single), (array == 3).astype(np.uint8))
    assert single.GetPixelID() == SimpleITK.sitkUInt8 and same_grid(single, labels)


def test_a_model_archive_with_the_folder_inside_another_folder(tmp_path, monkeypatch):
    """The downloader moves the model folder into place from wherever the archive holds it and leaves nothing else."""
    import io
    import zipfile
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("wrapper/Dataset999_Zipped/plans.json", "{}")
        z.writestr("__MACOSX/junk", "")
    response = SimpleNamespace(status_code=200, headers={"Content-Length": str(len(archive.getvalue()))},
                               iter_content=lambda chunk_size: [archive.getvalue()])
    monkeypatch.setattr(models.requests, "get", lambda url, stream: response)
    model = SimpleNamespace(model_identifier="clin_ct_zipped", folder_name="Dataset999_Zipped",
                            url="https://example.org/clin_ct_zipped.zip", base_directory=str(tmp_path / "models"),
                            directory=str(tmp_path / "models" / "Dataset999_Zipped"))
    models.Model._Model__download(model, system.OutputManager(False, False))
    assert sorted(os.listdir(tmp_path / "models")) == ["Dataset999_Zipped"]
    assert sorted(os.listdir(model.directory)) == ["model_version.json", "plans.json"]
    model.folder_name, model.directory = "Dataset998_Missing", str(tmp_path / "models" / "Dataset998_Missing")
    with pytest.raises(FileNotFoundError, match="does not hold the folder"):
        models.Model._Model__download(model, system.OutputManager(False, False))


def test_a_failed_roi_model_fails_again_with_its_own_error(tiny_model, tmp_path, monkeypatch):
    """A model that failed keeps its input: asked again (as a requested model after it failed as a roi model), it
    fails with the error of the model, not with a missing input."""
    roi_model, with_roi = register_with_roi(monkeypatch, tmp_path)
    with streaming.MoosePredictor([with_roi, roi_model], "cpu") as predictor:
        def broken(*args, **kwargs):
            raise RuntimeError("out of memory")
        monkeypatch.setattr(predictor.predictors[roi_model], "predict_from_data_iterator_streaming", broken)
        (result,) = predictor.predict([synthetic_ct()], on_error="skip")
    assert all("out of memory" in result[model].error for model in (with_roi, roi_model))


def test_the_nibabel_reorienting_reader_and_its_simpleitk_twin_give_the_same_case(tmp_path, monkeypatch):
    """A model whose plans name NibabelIOWithReorient runs with SimpleITKIOWithReorient: same arrays and spacing, in
    every storage orientation."""
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    for code in ("LAS", "RAS", "LPS", "RPI"):
        file = str(tmp_path / f"{code}.nii.gz")
        SimpleITK.WriteImage(SimpleITK.DICOMOrient(synthetic_ct(), code), file)
        expected_array, expected_properties = NibabelIOWithReorient().read_images([file])
        array, properties = streaming.image_to_case(SimpleITK.ReadImage(file), "SimpleITKIOWithReorient")
        np.testing.assert_array_equal(array, expected_array)
        assert array.dtype == expected_array.dtype
        np.testing.assert_allclose(properties["spacing"], expected_properties["spacing"], rtol=1e-6)
    identifier = register(monkeypatch, tmp_path, plans_reader="NibabelIOWithReorient")
    assert models.Model(identifier, system.OutputManager(False, False)).reader == "SimpleITKIOWithReorient"


def test_a_model_moosez_cannot_read_images_for_fails_when_it_is_loaded(tmp_path, monkeypatch):
    identifier = register(monkeypatch, tmp_path, plans_reader="NibabelIO")
    with pytest.raises(NotImplementedError, match="NibabelIO"):
        models.Model(identifier, system.OutputManager(False, False))


def test_a_model_named_twice_runs_once(tiny_model):
    with streaming.MoosePredictor([tiny_model, tiny_model], "cpu") as predictor:
        assert [model.model_identifier for model in predictor.requested] == [tiny_model]
        (result,) = predictor.predict([synthetic_ct()])
    assert isinstance(result[tiny_model], SimpleITK.Image)


def test_single_label_files_of_one_folder_wait_for_the_model_before(tiny_model, tmp_path):
    """Models of one subject share its -sl folder; their exports write into it one after the other, in model order."""
    image = SimpleITK.DICOMOrient(synthetic_ct(), "RAS")
    array, properties = streaming.image_to_case(image, "SimpleITKIOWithReorient")
    labels = (np.abs(array[0]).astype(np.int64) % 3).astype(np.uint8)
    model = models.Model(tiny_model, system.OutputManager(False, False))
    folder = tmp_path / "single"
    before = Future()
    with ThreadPoolExecutor(1) as writer:
        export = writer.submit(streaming.export_case, labels, properties, model,
                               {"single_labels_directory": str(folder)}, None, before)
        with pytest.raises(TimeoutError):
            export.result(timeout=0.5)
        assert not folder.exists()
        before.set_exception(RuntimeError("the model before failed"))      # its outcome does not matter
        export.result()
    assert sorted(f.name for f in folder.iterdir()) == ["liver.nii.gz", "spleen.nii.gz"]


def test_run_subjects_writes_each_subjects_information(tiny_model, tmp_path):
    """Statistics are opt-in: without them no stats folder and no statistics entries."""
    subjects = []
    for seed in range(2):
        subject = tmp_path / f"subject{seed}"
        subject.mkdir()
        SimpleITK.WriteImage(synthetic_ct(seed), str(subject / f"CT_subject{seed}.nii.gz"))
        subjects.append(str(subject))
    directories = {subject: file_utilities.moose_folder_structure(subject) for subject in subjects}
    streaming.run_subjects(subjects, [tiny_model], directories, "cpu")
    for seed, subject in enumerate(subjects):
        moose_directory, segmentations_directory, stats_directory = directories[subject]
        information = json.load(open(os.path.join(moose_directory, "MOOSE_information.json")))
        assert set(information["models"][tiny_model]) == {"model_directory", "segmentation_file"}
        assert information["models"][tiny_model]["segmentation_file"] == \
            os.path.join(segmentations_directory, f"clin_CT_tiny_segmentation_CT_subject{seed}.nii.gz")
        assert os.path.isfile(information["models"][tiny_model]["segmentation_file"])
        assert not os.path.exists(stats_directory)


def test_moosez_installs_a_model_archive_into_a_models_directory_that_does_not_exist_yet(tmp_path, monkeypatch):
    """moosez -mi: the model folder goes into the models directory (created if missing) wherever the archive holds it,
    the registry entry then finds and runs it; an archive without a known model folder installs nothing."""
    import shutil
    make_tiny_model(str(tmp_path / "packed" / "wrapper"))
    archive = shutil.make_archive(str(tmp_path / "tiny_model"), "zip", str(tmp_path / "packed"))
    models_directory = str(tmp_path / "site-packages" / "moosez" / "models" / "nnunet_trained_models")
    monkeypatch.setattr(system, "MODELS_DIRECTORY_PATH", models_directory)
    monkeypatch.setitem(models.MODEL_METADATA, TINY_MODEL, {
        constants.KEY_URL: f"{constants.LOCAL_URL_PREFIX}tiny", constants.KEY_FOLDER_NAME: TINY_FOLDER})

    assert models.install_model_archive(archive, system.OutputManager(False, False)) == [TINY_MODEL]
    assert os.listdir(models_directory) == [TINY_FOLDER]
    assert not os.path.exists(os.path.join(models_directory, TINY_FOLDER, "model_version.json"))   # local entry
    (label_map,) = predict(TINY_MODEL, [synthetic_ct()])
    assert isinstance(label_map, SimpleITK.Image)

    shutil.make_archive(str(tmp_path / "unknown"), "zip", str(tmp_path / "packed" / "wrapper" / TINY_FOLDER))
    assert models.install_model_archive(str(tmp_path / "unknown.zip"), system.OutputManager(False, False)) == []
    assert os.listdir(models_directory) == [TINY_FOLDER]


def test_moosez_files_builds_and_downloads_the_models_before_the_timed_prediction(tiny_model, tmp_path, monkeypatch):
    """moosez -i: every model is built (downloaded if needed) before the spinner starts and the prediction is timed, so
    neither the download progress nor its time ends up in the prediction; missing weights end the run with a message."""
    from moosez import moosez as cli
    events = []
    build = models.Model.__init__
    monkeypatch.setattr(models.Model, "__init__", lambda self, *a, **k: (events.append("model"), build(self, *a, **k))[1])
    monkeypatch.setattr(system.OutputManager, "spinner_start", lambda self, text: events.append("spinner"))
    monkeypatch.setattr(system, "check_device", lambda output_manager=None: ("cpu", None))
    SimpleITK.WriteImage(synthetic_ct(), str(tmp_path / "case.nii.gz"))
    cli.moose_files([str(tmp_path / "case.nii.gz")], str(tmp_path / "out"), [tiny_model], system.OutputManager(False, False))
    assert "model" in events and events.index("spinner") > max(i for i, e in enumerate(events) if e == "model")
    assert (tmp_path / "out" / "clin_CT_tiny_segmentation_case.nii.gz").is_file()

    monkeypatch.setitem(models.MODEL_METADATA, "clin_ct_missing", {
        constants.KEY_URL: f"{constants.LOCAL_URL_PREFIX}missing", constants.KEY_FOLDER_NAME: "Dataset998_Missing"})
    with pytest.raises(SystemExit):
        cli.prepare_models(["clin_ct_missing"], system.OutputManager(False, False))
