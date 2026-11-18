#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The moosez inference pipeline. Every model is an nnU-Net model and runs through nnU-Net's streaming predictor:

    model = { nnU-Net model folder, reader, input resampling, roi: { model, crop_labels, keep_label } or none }
                                                                                              (models.MODEL_METADATA)
    segment(model, image):
        slab = the slices that hold crop_labels in segment(roi model, image)                  # only models with a roi
        seg  = nnU-Net on image[slab]: the model's preprocessing, sliding window, the logits resampled to the input
               grid, argmax (nnUNetPredictorStreaming)
        seg -> pasted into the full grid (the slices of keep_label)

    per image:  read once -> segment(model, image) for every requested model -> label map, -sl files, statistics

The label maps come out on the input image's grid and in its orientation: nnU-Net resamples the logits, so moosez
neither resamples label maps nor splits images into chunks. What moosez adds to nnU-Net is in this file: the image <-> array adapters
(the model's reader brings every image into the orientation the model was trained in, and the label map back), the
cubic input resampling of the models that were trained on images moosez had resampled (resample_cubic), the roi step
and the outputs (export_case).

All models of a call are loaded once (MoosePredictor), while the first images are read. Threads of this process read
and preprocess the next images and write the outputs of the finished ones while the GPU predicts; SimpleITK, numpy and
torch release the GIL.
"""

import functools
import itertools
import os
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from threading import BoundedSemaphore, Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK
import torch
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming
from nnunetv2.inference.streaming_pipeline import CaseFailure, load_and_preprocess, load_npy_case
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
from scipy import ndimage

from moosez import constants
from moosez import file_utilities
from moosez import image_processing
from moosez import models
from moosez import system

IMAGE_FILE_ENDINGS = ('.nii.gz', '.nii', '.mha', '.nrrd')
# subjects read and preprocessed ahead of the GPU, and threads for reading and for writing
READ_AHEAD = 2
NUM_THREADS = 3
# threads that write the -sl files of one label map (each holds a mask of the image's size)
SINGLE_LABEL_THREADS = 4


# ------------------------------------------------------------------ image <-> nnU-Net arrays ------------------------
@functools.lru_cache(maxsize=None)
def _reader_class(reader: str):
    """The nnU-Net reader / writer class of that name (looked up once: the lookup scans nnU-Net's imageio package)."""
    return recursive_find_reader_writer_by_name(reader)


def image_to_case(image: SimpleITK.Image, reader: str, orientation: str = "RAS") -> Tuple[np.ndarray, dict]:
    """
    The (1, z, y, x) array and the properties the reader returns for this image stored as a file (nnU-Net's
    read_sitk_images), so files and in-memory images take the same path. A reorienting reader (SimpleITKIOWithReorient)
    brings any input into orientation, the one the model was trained in, and its writer back out of it
    (case_to_image).
    """
    if image.GetDimension() != 3:
        raise ValueError(f"moosez segments 3d images, this one has {image.GetDimension()} dimensions")
    if reader == 'SimpleITKIOWithReorient':
        return _reader_class(reader)().read_sitk_images([image], orientation=orientation)
    return _reader_class(reader)().read_sitk_images([image])


def case_to_image(segmentation: np.ndarray, properties: dict, reader: str) -> SimpleITK.Image:
    """The label map as an image on the input's grid and in its orientation (inverse of image_to_case)."""
    return _reader_class(reader)().seg_to_sitk(segmentation, properties)


class Case:
    """An image during a call: read once, brought into each reader's arrays once; per model the preprocessed input
    (until the model has run) and, for the models that give a roi, the label map."""

    def __init__(self, image: SimpleITK.Image):
        self.image = image
        self.arrays, self.preprocessed, self.segmentations = {}, {}, {}

    def array(self, model: models.Model) -> Tuple[Optional[np.ndarray], dict]:
        """The image as the model's reader hands it to nnU-Net (the array is None once released, see release)."""
        key = (model.reader, model.orientation)
        if key not in self.arrays:
            self.arrays[key] = image_to_case(self.image, *key)
        return self.arrays[key]

    def release(self, keep_arrays: set, keep_image: bool):
        """Drops the voxels the rest of the call does not need, so that they are not held through the GPU passes: the
        reader arrays not in keep_arrays (their properties stay, the export needs them) and, unless keep_image, the
        image."""
        self.arrays = {key: (array if key in keep_arrays else None, properties)
                       for key, (array, properties) in self.arrays.items()}
        if not keep_image:
            self.image = None


# ------------------------------------------------------------------ cubic input resampling --------------------------
@functools.lru_cache(maxsize=16)
def zoom_operator(n_in: int, n_out: int) -> torch.Tensor:
    """
    scipy.ndimage.zoom along one axis as a matrix (n_out, n_in): cubic B-spline with its prefilter, voxel centres
    aligned (grid_mode=True: the grid of nnU-Net's own resampling, on which the logits go back), edge values beyond
    the image. zoom is linear and separable, so zooming an identity matrix along one axis is its exact operator.
    """
    operator = ndimage.zoom(np.eye(n_in), (1, n_out / n_in), order=3, mode='nearest', grid_mode=True).T
    assert operator.shape == (n_out, n_in)
    return torch.from_numpy(operator).float()


def resample_cubic(data: Union[np.ndarray, torch.Tensor], new_shape: Sequence[int], current_spacing=None,
                   new_spacing=None, device: Union[str, torch.device] = 'cpu') -> np.ndarray:
    """
    An input resampling function for nnU-Net's preprocessing (resampling_fn_data): data (c, x, y, z) to
    (c, *new_shape), cubic on every axis, as matrix products with zoom_operator on device. The axes go in the order of
    their reduction and the first one in pieces, so the image at its full resolution never sits on the device.
    """
    data = torch.as_tensor(data)
    for axis in sorted(range(len(new_shape)), key=lambda a: new_shape[a] / data.shape[a + 1]):
        if data.shape[axis + 1] != new_shape[axis]:
            operator = zoom_operator(data.shape[axis + 1], int(new_shape[axis])).to(device).T
            data = data.movedim(axis + 1, -1)
            data = torch.cat([piece.to(device) @ operator for piece in data.split(32, dim=1)], dim=1)
            data = data.movedim(-1, axis + 1)
    return data.contiguous().cpu().numpy()      # contiguous on the device, not reordered on the host afterwards


class CubicInput(ConfigurationManager):
    """A model's nnU-Net configuration with resample_cubic as the input resampling (nnU-Net finds the function the
    plans name only in its own package, so the plans cannot name this one)."""

    def __init__(self, configuration: dict, device: torch.device):
        super().__init__(configuration)
        self.device = device

    @property
    def resampling_fn_data(self) -> Callable:
        return functools.partial(resample_cubic, device=self.device)


# ------------------------------------------------------------------ roi ---------------------------------------------
def label_slab(segmentation: np.ndarray, labels: Sequence[int], largest_component: bool = False) -> \
        Optional[Tuple[int, int]]:
    """
    The range of slices (first, last + 1) along the first axis of the label map that hold any of labels; None if none
    does. The first array axis is the body axis for the models read in RAS. largest_component: the slices of the
    largest connected component of those labels.
    """
    wanted = np.zeros(max(int(segmentation.max()), *labels) + 1, dtype=bool)
    wanted[list(labels)] = True
    slices = np.flatnonzero(wanted[segmentation].any(axis=(1, 2)))
    if len(slices) == 0:
        return None
    first, last = int(slices[0]), int(slices[-1]) + 1
    if largest_component:
        mask = SimpleITK.GetImageFromArray(wanted[segmentation[first:last]].astype(np.uint8))
        largest = SimpleITK.RelabelComponent(SimpleITK.ConnectedComponent(mask), sortByObjectSize=True) == 1
        slices = np.flatnonzero(SimpleITK.GetArrayViewFromImage(largest).any(axis=(1, 2)))
        first, last = first + int(slices[0]), first + int(slices[-1]) + 1
    return first, last


# ------------------------------------------------------------------ outputs -----------------------------------------
def write_single_labels(label_image: SimpleITK.Image, organ_indices: Dict[int, str], output_directory: str) -> None:
    """One binary file per label present (moosez -sl), written by threads from the label map in memory."""
    os.makedirs(output_directory, exist_ok=True)
    present = [label for label in image_processing.label_voxel_counts(label_image) if label in organ_indices]

    def write(label: int):
        SimpleITK.WriteImage(SimpleITK.BinaryThreshold(label_image, label, label, 1, 0),
                             os.path.join(output_directory, f"{organ_indices[label]}.nii.gz"))

    with ThreadPoolExecutor(SINGLE_LABEL_THREADS) as executor:
        list(executor.map(write, present))


def write_statistics(label_image: SimpleITK.Image, model: models.Model, volume_csv: str,
                     images: Dict[str, SimpleITK.Image], intensity_csvs: Dict[str, str]) -> None:
    """
    The statistics files of a label map: volumes, and intensities for each image in images ({modality: image}) that
    has a csv in intensity_csvs. For the image of another modality (PET for a CT label map) the label map is resliced
    onto that image's grid with nearest neighbour.
    """
    os.makedirs(os.path.dirname(os.path.abspath(volume_csv)), exist_ok=True)
    image_processing.get_shape_statistics(label_image, model, volume_csv)
    for modality, csv_file in intensity_csvs.items():
        image = images[modality]
        labels = label_image if modality == model.modality_full else \
            image_processing.reslice_label_image(image, label_image)
        image_processing.get_intensity_statistics(image, labels, model, csv_file)


def export_case(segmentation: np.ndarray, properties: dict, model: models.Model, output: dict,
                images: Optional[Dict[str, SimpleITK.Image]] = None, after: Optional[Future] = None):
    """
    The outputs of one label map (runs in a writer thread). segmentation and properties: in the arrays of the model's
    reader (image_to_case). output: {'file': path of the label map, 'organ_indices_directory': folder for the model's
    <prefix>organ_indices.json, 'single_labels_directory': folder for -sl files, 'statistics': {'volume_csv': path,
    'intensity': {modality: csv path}}}, each optional; images: the subject's images for the intensity statistics,
    {modality: image}; after: the export whose -sl files go to the same folder, which writes its files first (models
    with labels of the same name: the last model's file stays, as in moosez 3.2.2). Returns the path of the label map
    if it was written, else the label map as an image.
    """
    image = case_to_image(segmentation, properties, model.reader)
    if output.get('file'):
        SimpleITK.WriteImage(image, output['file'], True)
    if output.get('organ_indices_directory'):
        model.organ_indices_to_json(output['organ_indices_directory'])
    if output.get('single_labels_directory'):
        if after is not None:
            wait([after])
        write_single_labels(image, model.organ_indices, output['single_labels_directory'])
    statistics = output.get('statistics')
    if statistics:
        write_statistics(image, model, statistics['volume_csv'], images, statistics['intensity'])
    return output.get('file') or image


# ------------------------------------------------------------------ the pipeline ------------------------------------
class MoosePredictor:
    """
    The models of a call, loaded once; predict() takes any number of subjects, as often as needed. The networks are built
    and their weights loaded at the first use (predictors), which predict() reaches after it has started to read its
    first images: reading and loading overlap.

        with MoosePredictor(['clin_ct_organs', 'clin_ct_body_composition'], 'cuda') as moose:
            label_images = moose.predict(['ct_1.nii.gz', 'ct_2.nii.gz'])       # per image {model identifier: image}

    torch.compile follows nnU-Net's 'auto' policy (on for calls with more than a few subjects) unless compile is set.
    """

    def __init__(self, model_identifiers: Union[str, models.Model, Sequence[Union[str, models.Model]]] = constants.DEFAULT_MODEL,
                 accelerator: Optional[str] = None, compile: Union[bool, str] = 'auto',
                 output_manager: Optional[system.OutputManager] = None):
        if output_manager is None:
            output_manager = system.OutputManager(False, False)
        if accelerator is None:
            accelerator, _ = system.check_device(output_manager)
        self.device = torch.device(accelerator)
        if isinstance(model_identifiers, (str, models.Model)):
            model_identifiers = [model_identifiers]
        requested = [m if isinstance(m, models.Model) else models.Model(m, output_manager)
                     for m in dict.fromkeys(model_identifiers)]
        self.requested = list({model.model_identifier: model for model in requested}.values())  # a model named twice runs once
        self.models = models.with_roi_models(self.requested, output_manager)
        # the label maps of these models are another model's roi
        self._roi_models = {model.roi['model'] for model in self.models.values() if model.roi is not None}
        for model in self.models.values():
            if model.roi is not None:
                roi = self.models[model.roi['model']]
                if (roi.reader, roi.orientation, roi.modality_full) != (model.reader, model.orientation, model.modality_full):
                    raise ValueError(f"{model} and the model of its roi, {roi}, must read the same images the same way")
        self._compile, self._output_manager = compile, output_manager
        self._predictors, self._predictors_lock = None, Lock()

    @property
    def predictors(self) -> Dict[str, nnUNetPredictorStreaming]:
        """The nnU-Net predictor of every model, built at the first use (by whichever thread asks first; the others
        wait). The networks are built here; their weights load in the background and the first prediction waits for
        them."""
        with self._predictors_lock:
            if self._predictors is None:
                predictors = {}
                with self._output_manager.manage_nnUNet_output():
                    for identifier, model in self.models.items():
                        cubic = model.input_resampling == constants.INPUT_RESAMPLING_CUBIC
                        predictor = nnUNetPredictorStreaming(
                            tile_step_size=0.5, use_gaussian=True, use_mirroring=True, device=self.device,
                            verbose=False, verbose_preprocessing=False, allow_tqdm=False, compile=self._compile,
                            quiet=True, force_trilinear=cubic)
                        predictor.initialize_from_folder_base_trainer(
                            model.configuration_directory, use_folds=model.folds, load_weights_in_background=True)
                        if cubic:
                            predictor.configuration_manager = CubicInput(
                                predictor.configuration_manager.configuration, self.device)
                        predictors[identifier] = predictor
                self._predictors = predictors
            return self._predictors

    def close(self):
        """Releases the models; the predictor cannot be used afterwards."""
        with self._predictors_lock:
            self._predictors = {}
        empty_cache(self.device)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def preprocess(self, model: models.Model, array: np.ndarray, properties: dict) -> Tuple[np.ndarray, dict]:
        """nnU-Net's preprocessing of the model for an image in the arrays of the model's reader."""
        predictor = self.predictors[model.model_identifier]
        return load_and_preprocess(load_npy_case, (array, properties), predictor.plans_manager,
                                   predictor.configuration_manager, predictor.dataset_json,
                                   predictor.configuration_manager.preprocessor_class(verbose=False))

    def segment(self, model: models.Model, case: Case, num_cases: int = 1) -> Optional[np.ndarray]:
        """
        The label map of the model for the image, in the arrays of the model's reader; None if the image does not
        hold the structures that give the model's roi. num_cases: the number of subjects of the call (nnU-Net compiles
        the networks for larger ones).
        """
        identifier = model.model_identifier
        if identifier in case.segmentations:
            return case.segmentations[identifier]

        def predict(data: np.ndarray, properties: dict) -> np.ndarray:
            return self.predictors[identifier].predict_from_data_iterator_streaming(
                [{'data': data, 'data_properties': properties, 'ofile': None}], num_cases=num_cases)[0]

        if model.roi is None:
            segmentation = predict(*case.preprocessed[identifier])
            del case.preprocessed[identifier]
        else:
            array, properties = case.array(model)
            roi = self.segment(self.models[model.roi['model']], case, num_cases)
            crop = None if roi is None else label_slab(roi, model.roi['crop_labels'])
            segmentation = None
            if crop is not None:
                # keep_label is one of crop_labels, so its slices (first, last within the crop) lie inside the crop
                first, last = label_slab(roi[crop[0]:crop[1]], [model.roi['keep_label']], largest_component=True) or \
                    (0, crop[1] - crop[0])
                cropped = predict(*self.preprocess(model, array[:, crop[0]:crop[1]], properties))
                segmentation = np.zeros(array.shape[1:], dtype=cropped.dtype)
                segmentation[crop[0] + first:crop[0] + last] = cropped[first:last]
        if identifier in self._roi_models:      # runs once per image, whichever model asks for it first
            case.segmentations[identifier] = segmentation
        return segmentation

    def _read(self, subject: Dict[str, Union[str, SimpleITK.Image]], keep_images: bool) -> Dict[str, Case]:
        """A reader thread's work for one subject: each image read once, and for every model that runs on the whole
        image its reader's arrays and its preprocessing. Returns {modality: Case}."""
        cases, by_source = {}, {}
        for modality, source in subject.items():
            key = source if isinstance(source, str) else id(source)
            if key not in by_source:
                by_source[key] = Case(image_processing.image_read(source) if isinstance(source, str) else source)
            cases[modality] = by_source[key]
        for identifier, model in self.models.items():
            case = cases.get(model.modality_full)
            if case is not None and model.roi is None:
                case.preprocessed[identifier] = self.preprocess(model, *case.array(model))
        # what is left to hold: the reader arrays that models with a roi crop, the images for intensity statistics
        for case in by_source.values():
            case.release({(model.reader, model.orientation) for model in self.models.values()
                          if model.roi is not None and cases.get(model.modality_full) is case}, keep_images)
        return cases

    def predict(self, subjects: Sequence[Union[str, SimpleITK.Image, Dict[str, Union[str, SimpleITK.Image]]]],
                outputs: Optional[Sequence[Dict[str, dict]]] = None, on_error: str = 'raise',
                progress: Optional[Callable[[int, str], None]] = None) -> List[Dict[str, Any]]:
        """
        Segments the subjects with every requested model. A subject is an image (file path or SimpleITK image), given
        to every model, or {modality: image} with the modalities as models.Model.modality_full ('CT', 'PT_FDG', ...),
        where a model runs if its modality is there. outputs: per subject {model identifier: export_case output};
        default: the label maps are returned as SimpleITK images on the inputs' grids and in their orientations.
        Returns per subject {model identifier: export_case's result, or None if the image does not hold the
        structures of the model's roi}. on_error 'skip': a failing subject or model gives an nnU-Net CaseFailure in
        place of its result and the others carry on. progress(subject index, model identifier) is called after each
        GPU pass.
        """
        subjects = [s if isinstance(s, dict) else {model.modality_full: s for model in self.requested}
                    for s in subjects]
        outputs = outputs if outputs is not None else [{} for _ in subjects]
        results = [{} for _ in subjects]
        keep_images = any(output.get('statistics') for subject in outputs for output in subject.values())

        def failure(index: int, stage: str, error: Exception) -> CaseFailure:
            if on_error == 'raise':
                raise error
            return CaseFailure.from_exception(index, stage, error)

        waiting = BoundedSemaphore(NUM_THREADS + 2)                 # label maps that wait for a writer thread
        with ThreadPoolExecutor(NUM_THREADS, 'moosez_reader') as readers, \
                ThreadPoolExecutor(NUM_THREADS, 'moosez_writer') as writers:
            todo = iter(subjects)
            read = lambda subject: readers.submit(self._read, subject, keep_images)
            reading = deque(read(subject) for subject in itertools.islice(todo, READ_AHEAD))
            self.predictors                             # the models load while the first images are read
            for index in range(len(subjects)):
                reading.extend(read(subject) for subject in itertools.islice(todo, 1))
                try:
                    cases = reading.popleft().result()
                except Exception as e:
                    error = failure(index, 'load', e)
                    results[index] = {model.model_identifier: error for model in self.requested
                                      if model.modality_full in subjects[index]}
                    continue
                images = {modality: case.image for modality, case in cases.items()}
                single_label_exports = {}           # -sl folder: the last export that writes into it
                for model in self.requested:
                    case, identifier = cases.get(model.modality_full), model.model_identifier
                    if case is None:
                        continue
                    try:
                        segmentation = results[index][identifier] = self.segment(model, case, len(subjects))
                        if segmentation is not None:
                            output = outputs[index].get(identifier, {})
                            directory = output.get('single_labels_directory')
                            waiting.acquire()
                            results[index][identifier] = writers.submit(
                                export_case, segmentation, case.array(model)[1], model, output, images,
                                single_label_exports.get(directory))
                            results[index][identifier].add_done_callback(lambda _: waiting.release())
                            if directory:
                                single_label_exports[directory] = results[index][identifier]
                    except Exception as e:
                        results[index][identifier] = failure(index, 'predict', e)
                    if progress is not None:
                        progress(index, identifier)
                del cases, images, single_label_exports
        for index, result in enumerate(results):
            for identifier, value in result.items():
                if isinstance(value, Future):
                    try:
                        result[identifier] = value.result()
                    except Exception as e:
                        result[identifier] = failure(index, 'export', e)
        return results


# ------------------------------------------------------------------ runs with files (CLI) ---------------------------
def segmentation_file(model: models.Model, directory: str, stem: str) -> str:
    """The file of a model's label map for the image named stem."""
    return os.path.join(directory, f"{model.multilabel_prefix}segmentation_{stem}.nii.gz")


def _output(model: models.Model, directory: str, stem: str, single_labels_directory: Optional[str],
            stats_directory: Optional[str], stats_name: str, modalities: Sequence[str]) -> dict:
    """The export_case output of a model for one image: the label map file and, if asked for, -sl and statistics."""
    prefix = model.multilabel_prefix
    statistics = None if stats_directory is None else {
        'volume_csv': os.path.join(stats_directory, f"{prefix}{stats_name}_volume.csv"),
        'intensity': {m: os.path.join(stats_directory, f"{prefix}{stats_name}_{m}_intensity.csv") for m in modalities}}
    return {'file': segmentation_file(model, directory, stem), 'single_labels_directory': single_labels_directory,
            'statistics': statistics}


def _information(output: dict) -> dict:
    """The entry of a written label map in MOOSE_information.json."""
    information = {"segmentation_file": output['file']}
    if output['statistics'] is not None:
        information["VOL_stats_file"] = output['statistics']['volume_csv']
        for modality, csv_file in output['statistics']['intensity'].items():
            information[f"{modality}_intensity_stats_file"] = csv_file
    return information


def _run(moose: MoosePredictor, subjects: list, outputs: List[Dict[str, dict]], names: List[str],
         output_manager: system.OutputManager) -> List[Dict[str, dict]]:
    """predict with a progress spinner; reports what failed or had no roi. Returns per subject {model identifier:
    output} of the label maps written."""
    def progress(index: int, identifier: str):
        output_manager.spinner_update(f'[{index + 1}/{len(subjects)}] {names[index]}: {identifier} predicted')

    start = time.time()
    with output_manager.manage_nnUNet_output():
        results = moose.predict(subjects, outputs, on_error='skip', progress=progress)
    written = []
    for name, result, output in zip(names, results, outputs):
        written.append({identifier: output[identifier] for identifier, value in result.items() if isinstance(value, str)})
        for identifier, value in result.items():
            if isinstance(value, CaseFailure):
                output_manager.console_update(f"{constants.ANSI_RED} {identifier}: {name} failed ({value.stage}): "
                                              f"{value.error}{constants.ANSI_RESET}")
                output_manager.log_update(f"     - {identifier}: {name} failed in {value.stage}:\n{value.traceback}")
            elif value is None:
                output_manager.spinner_warn(f'{name}: organ to crop from not in initial FOV. No segmentation result '
                                            f'({identifier}) for this subject.')
                output_manager.log_update(f"     - {identifier}: {name}: organ to crop from not in initial FOV.")
    output_manager.log_update(f"   - {len(subjects)} subjects, {len(moose.requested)} models: predicted in "
                              f"{round(time.time() - start, 1)} s")
    return written


def run_subjects(subjects: List[str], requested_models: Sequence[Union[str, models.Model]],
                 moose_directories: Dict[str, Tuple[str, str, str]], accelerator: str,
                 output_manager: Optional[system.OutputManager] = None, single_labels: bool = False,
                 statistics: bool = False) -> None:
    """
    moosez -d: every subject folder with every requested model whose modality the subject has. Writes into
    <subject>/moosez-<timestamp>/ (moose_directories: file_utilities.moose_folder_structure) the label map and organ
    indices of each model, with statistics also its volume statistics and its intensity statistics for each of the
    requested models' modalities the subject has, and the subject's MOOSE_information.json.
    """
    if output_manager is None:
        output_manager = system.OutputManager(False, False)
    moose = MoosePredictor(requested_models, accelerator, output_manager=output_manager)
    inputs, outputs = [], []
    for subject in subjects:
        files = {}
        for model in moose.requested:
            modality_files = file_utilities.get_modality_NIFTI_files(subject, model.modality_full + "_")
            if modality_files:
                files[model.modality_full] = modality_files[0]
            else:
                output_manager.log_update(f"     - No {model.modality_full}_ image found in {subject}. Skipping {model}.")
        _, segmentations_directory, stats_directory = moose_directories[subject]
        inputs.append(files)
        outputs.append({model.model_identifier: {**_output(
            model, segmentations_directory, file_utilities.get_nifti_file_stem(files[model.modality_full]),
            os.path.join(segmentations_directory, constants.SINGLE_LABELS_FOLDER) if single_labels else None,
            stats_directory if statistics else None, os.path.basename(subject), list(files)),
            'organ_indices_directory': segmentations_directory}
            for model in moose.requested if model.modality_full in files})
    with moose:
        written = _run(moose, inputs, outputs, [os.path.basename(subject) for subject in subjects], output_manager)
    for subject, subject_outputs in zip(subjects, written):
        information = system.get_system_information()
        information["single_labels"] = single_labels
        information["models"] = {identifier: {"model_directory": moose.models[identifier].directory, **_information(output)}
                                 for identifier, output in subject_outputs.items()}
        system.write_information_json(information, moose_directories[subject][0])


def collect_image_files(inputs: Sequence[str]) -> List[str]:
    """Image files from files and folders (a folder contributes its image files, not its subfolders), in order."""
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files += sorted(file_utilities.get_files(path, '', IMAGE_FILE_ENDINGS))
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")
    return files


def image_file_stem(file_path: str) -> str:
    name = os.path.basename(file_path)
    for ending in IMAGE_FILE_ENDINGS:
        if name.endswith(ending):
            return name[:-len(ending)]
    return os.path.splitext(name)[0]


def run_files(image_files: List[str], output_directory: str, requested_models: Sequence[Union[str, models.Model]],
              accelerator: str, output_manager: system.OutputManager, single_labels: bool = False,
              statistics: bool = False) -> dict:
    """
    moosez -i ... -o ...: a flat list of image files, each with every requested model. Writes into output_directory,
    per model and image, <prefix>segmentation_<stem>.nii.gz, once per model <prefix>organ_indices.json, and with
    statistics stats/<prefix><stem>_volume.csv / _<modality>_intensity.csv. Returns the information for
    MOOSE_information.json.
    """
    stems = [image_file_stem(f) for f in image_files]
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise ValueError(f"Several input images share the name(s) {', '.join(duplicates)}; outputs would overwrite each other")
    os.makedirs(output_directory, exist_ok=True)
    stats_directory = os.path.join(output_directory, constants.STATS_FOLDER) if statistics else None
    moose = MoosePredictor(requested_models, accelerator, output_manager=output_manager)
    outputs = [{model.model_identifier: _output(
        model, output_directory, stem,
        os.path.join(output_directory, constants.SINGLE_LABELS_FOLDER, f"{model.multilabel_prefix}{stem}") if single_labels else None,
        stats_directory, stem, [model.modality_full]) for model in moose.requested} for stem in stems]
    for model in moose.requested:
        model.organ_indices_to_json(output_directory)
    with moose:
        written = _run(moose, image_files, outputs, stems, output_manager)
    information = {}
    for model in moose.requested:
        cases = {stem: {"image_file": image_file, **_information(file_outputs[model.model_identifier])}
                 for stem, image_file, file_outputs in zip(stems, image_files, written)
                 if model.model_identifier in file_outputs}
        information[model.model_identifier] = {"model_directory": model.directory, "cases": cases}
    return information
