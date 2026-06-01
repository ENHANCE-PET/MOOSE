import SimpleITK
import time

from typing import Dict, List, Optional, Set, Tuple, Union
from moosez import system
from moosez import models
from moosez import constants
from moosez import image_processing
from moosez import predict
from moosez.benchmarking.benchmark import PerformanceObserver


# ---------------------------------------------------------------------------
# WORKFLOW_REGISTRY
#
# Defines multi-step workflows. A workflow is an ordered list of steps.
# Each step has:
#   - model       : the model identifier to run
#   - role        : "crop_FOV"     → result is used to crop the FOV for the next step
#                   "segmentation" → result is the final segmentation, yielded to the caller
#   - fov_config  : only for "crop_FOV" steps, defines how the crop is applied
#
# Any model NOT listed here is implicitly a single-step segmentation workflow:
#   [{"model": model_identifier, "role": "segmentation"}]
#
# To add a new multi-step workflow, add an entry here. No other file needs to change.
# ---------------------------------------------------------------------------

WORKFLOW_REGISTRY: Dict[str, List[Dict]] = {
    "clin_ct_body_composition": [
        {
            "model": "clin_ct_fast_vertebrae",
            "role": "crop_FOV",
            "fov_intensities": [20, 24],
            "crop_label": 22,
            "largest_component_only": True,
        },
        {
            "model": "clin_ct_body_composition",
            "role": "segmentation",
        },
    ],
    "clin_ct_face": [
        {
            "model": "clin_ct_face",
            "role": "segmentation",
        },
    ],
}


class WorkflowStep:
    def __init__(self, model: models.Model, role: str, fov_config: Optional[Dict] = None):
        if role not in ("segmentation", "crop_FOV"):
            raise ValueError(f"Unknown workflow step role: '{role}'")
        self.model = model
        self.role = role
        self.fov_config = fov_config

    def __repr__(self) -> str:
        return f"WorkflowStep(model={self.model.model_identifier}, role={self.role})"


class Workflow:
    def __init__(self, model_identifier: str, output_manager: system.OutputManager):
        self.steps: List[WorkflowStep] = []
        self.required_modalities: Set[str] = set()

        workflow_steps = WORKFLOW_REGISTRY.get(model_identifier, [{"model": model_identifier, "role": "segmentation"}])

        for workflow_step in workflow_steps:
            model = models.Model(workflow_step["model"], output_manager)
            self.required_modalities.add(model.modality)

            fov_config: Optional[Dict] = None
            if workflow_step["role"] == "crop_FOV":
                fov_config = {"fov_intensities": workflow_step["fov_intensities"],
                              "crop_label": workflow_step["crop_label"],
                              "largest_component_only": workflow_step.get("largest_component_only", False)}

            self.steps.append(WorkflowStep(model, workflow_step["role"], fov_config))

        self.target_step: WorkflowStep = next(s for s in self.steps if s.role == "segmentation")
        self.target_model: models.Model = self.target_step.model
        self.initial_desired_spacing: Tuple = self.steps[0].model.voxel_spacing

    @property
    def fov_crop_step(self) -> Optional[WorkflowStep]:
        """Returns the fov_crop step if one exists, otherwise None."""
        return next((s for s in self.steps if s.role == "crop_FOV"), None)

    @property
    def input_modality(self) -> str:
        """The modality prefix needed to look up this workflow's input image."""
        return self.steps[0].model.modality_full + "_"

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return " -> ".join(s.model.model_identifier for s in self.steps)

    def __repr__(self) -> str:
        return f"Workflow({self})"


def construct_workflows(model_identifiers: Union[str, List[str]], output_manager: system.OutputManager) -> List[Workflow]:
    if isinstance(model_identifiers, str):
        model_identifiers = [model_identifiers]

    output_manager.log_update(" SETTING UP MODEL WORKFLOWS:")
    workflows: List[Workflow] = []

    for identifier in model_identifiers:
        output_manager.log_update(f" - {identifier}")
        workflows.append(Workflow(identifier, output_manager))

    workflows.sort(key=lambda w: w.initial_desired_spacing)
    return workflows


def inference(image: SimpleITK.Image, model: models.Model, accelerator: str, output_manager: system.OutputManager, performance_observer: PerformanceObserver, image_array_cache: Dict = None) -> SimpleITK.Image:
    if image_array_cache is not None and model.voxel_spacing in image_array_cache:
        image_array_resampled = image_array_cache[model.voxel_spacing]
    else:
        performance_observer.record_phase(f"Resampling Image: {'x'.join(map(str, model.voxel_spacing))}")
        resampling_time_start = time.time()
        image_array_resampled = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, constants.INTERPOLATION, model.voxel_spacing)
        output_manager.log_update(f' - Resampling at {"x".join(map(str, model.voxel_spacing))} took: {round((time.time() - resampling_time_start), 2)}s')
        performance_observer.time_phase()

        if image_array_cache is not None:
            image_array_cache[model.voxel_spacing] = image_array_resampled

    performance_observer.record_phase(f"Predicting: {model.model_identifier}")
    segmentation_array = predict.predict_from_array_by_iterator(image_array_resampled, model, accelerator, output_manager)
    performance_observer.time_phase()

    performance_observer.record_phase(f"Segmentation array to image")
    segmentation_image = SimpleITK.GetImageFromArray(segmentation_array)
    segmentation_image.SetSpacing(image_processing.reverse_axes(model.voxel_spacing))
    segmentation_image.SetOrigin(image.GetOrigin())
    segmentation_image.SetDirection(image.GetDirection())

    segmentation_image_resampled = image_processing.ImageResampler.resample_segmentation(image, segmentation_image)
    performance_observer.time_phase()

    return segmentation_image_resampled


def threshold():
    pass


def crop_fov(image: SimpleITK.Image, segmentation: SimpleITK.Image, fov_config: Dict) -> Optional[SimpleITK.Image]:
    fov_intensities = fov_config["fov_intensities"]

    mask = SimpleITK.BinaryThreshold(segmentation, lowerThreshold=fov_intensities[0], upperThreshold=fov_intensities[1])

    label_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask)
    if label_stats.GetNumberOfLabels() == 0:
        return None

    bbox = label_stats.GetBoundingBox(1)
    # bbox: (x_start, y_start, z_start, x_size, y_size, z_size)
    # Crop only z, keep full x/y
    size = list(image.GetSize())
    size[2] = bbox[5]
    index = [0, 0, bbox[2]]

    return SimpleITK.RegionOfInterest(image, size=size, index=index)


def restrict_fov(segmentation: SimpleITK.Image, crop_segmentation: SimpleITK.Image, fov_config: Dict) -> SimpleITK.Image:
    crop_label = fov_config["crop_label"]
    largest_component_only = fov_config.get("largest_component_only", False)

    mask = SimpleITK.BinaryThreshold(crop_segmentation, lowerThreshold=crop_label, upperThreshold=crop_label)

    if largest_component_only:
        cc = SimpleITK.ConnectedComponent(mask)
        cc = SimpleITK.RelabelComponent(cc, sortByObjectSize=True)
        mask = SimpleITK.BinaryThreshold(cc, lowerThreshold=1, upperThreshold=1)

    label_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask)
    if label_stats.GetNumberOfLabels() == 0:
        return segmentation
    bbox = label_stats.GetBoundingBox(1)

    z_band = SimpleITK.Image(segmentation.GetSize(), SimpleITK.sitkUInt8)
    z_band.CopyInformation(segmentation)
    z_band[:, :, bbox[2]:bbox[2] + bbox[5]] = 1

    return SimpleITK.Mask(segmentation, z_band)


def run(images: Dict[str, SimpleITK.Image], workflow: Workflow, accelerator: str, output_manager: system.OutputManager, performance_observer: PerformanceObserver, image_array_caches: Dict = None) -> Optional[SimpleITK.Image]:
    crop_step = workflow.fov_crop_step

    if crop_step is None:
        input_modality = workflow.target_model.modality_full + "_"
        image = images[input_modality]
        if image_array_caches:
            image_array_cache = image_array_caches.get(input_modality, {})
        else:
            image_array_cache = None
        return inference(image, workflow.target_model, accelerator, output_manager, performance_observer, image_array_cache)

    crop_modality = crop_step.model.modality_full + "_"
    crop_image = images[crop_modality]
    crop_segmentation = inference(crop_image, crop_step.model, accelerator, output_manager, performance_observer)

    target_modality = workflow.target_model.modality_full + "_"
    target_image = images.get(target_modality, crop_image)
    cropped_image = crop_fov(target_image, crop_segmentation, crop_step.fov_config)
    if cropped_image is None:
        return None

    segmentation = inference(cropped_image, workflow.target_model, accelerator, output_manager, performance_observer)
    segmentation = image_processing.ImageResampler.resample_segmentation(target_image, segmentation)
    segmentation = restrict_fov(segmentation, crop_segmentation, crop_step.fov_config)

    return segmentation


def run_all(images: Dict[str, SimpleITK.Image], workflows: List[Workflow], output_manager: system.OutputManager, performance_observer: PerformanceObserver, accelerator: str, subjects_information: Tuple[str, int, int]):
    performance_observer.metadata_image_size = next(iter(images.values())).GetSize()
    subject_name, subject_index, number_of_subjects = subjects_information
    image_array_caches = {modality: {} for modality in images}

    for workflow in workflows:
        if workflow.input_modality not in images:
            output_manager.spinner_warn(f'[{subject_index + 1}/{number_of_subjects}] {subject_name}: no {workflow.input_modality} image available. Skipping {workflow.target_model}.')
            output_manager.log_update(f"     - No {workflow.input_modality} image found. Skipping {workflow.target_model}.")
            continue

        workflow_time_start = time.time()
        output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Running workflow for {subject_name} using {workflow}...')
        output_manager.log_update(f'   - Workflow {workflow}')

        segmentation = run(images, workflow, accelerator, output_manager, performance_observer, image_array_caches)

        if segmentation is None:
            output_manager.spinner_warn(f'[{subject_index + 1}/{number_of_subjects}] {subject_name}: organ to crop from not in initial FOV. No segmentation result ({workflow.target_model}) for this subject.')
            output_manager.log_update("     - Organ to crop from not in initial FOV.")
            continue

        output_manager.log_update(f"     - Workflow complete for {workflow.target_model} within {round((time.time() - workflow_time_start) / 60, 1)} min.")
        yield segmentation, workflow.target_model

