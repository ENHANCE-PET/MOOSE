import os
import json
import zipfile
import requests
import shutil
import tempfile
from typing import Tuple, List, Dict, Sequence
from moosez import system
from moosez.constants import (KEY_FOLDER_NAME, KEY_URL, KEY_READER, KEY_ORIENTATION, KEY_INPUT_RESAMPLING, KEY_ROI,
                              INPUT_RESAMPLING_CUBIC, FILE_NAME_DATASET_JSON, FILE_NAME_PLANS_JSON, ANSI_GREEN,
                              ANSI_RESET, ALLOWED_MODALITY_CONFIGURATIONS, LOCAL_URL_PREFIX, SUPPORTED_READERS,
                              SIMPLEITK_READER_FOR)
from moosez.mappings import SNOMED


# Every model is an nnU-Net model folder and runs through nnU-Net's streaming predictor (moosez/streaming.py) the way
# its plans say - reader (orientation; constants.SIMPLEITK_READER_FOR), preprocessing, resampling - unless its entry
# says otherwise:
#   KEY_READER             the nnU-Net reader / writer to use instead of the plans' one
#   KEY_ORIENTATION        the orientation a reorienting reader brings the images into (default: RAS)
#   KEY_INPUT_RESAMPLING   INPUT_RESAMPLING_CUBIC: moosez resamples the input to the model's spacing (cubic B-spline on
#                          all axes, on the model's device) instead of the plans' function, and the logits go back
#                          trilinear
#   KEY_ROI                {"model", "crop_labels", "keep_label"}: the model runs on the slices that hold crop_labels in
#                          the label map of "model", and only the slices of keep_label (its largest connected
#                          component) are kept in the output
#
# RAS_CUBIC marks the models trained on images that moosez had brought into RAS and resampled itself (cubic B-spline,
# no separate treatment of the slice axis); their plans (SimpleITKIO, nnU-Net's default resampling) do not say so.
RAS_CUBIC = {KEY_READER: "SimpleITKIOWithReorient", KEY_INPUT_RESAMPLING: INPUT_RESAMPLING_CUBIC}
# Four of them (KEY_ORIENTATION: "LAS") give left / right swapped label maps on RAS arrays and label maps that agree
# with the newer models on LAS arrays, so they get LAS. The models that mirror at test time do not depend on it.

MODEL_METADATA = {
    # One model for the 125 structures of the ENHANCE.PET 1.6k release (the six clin_ct_{cardiac, muscles, organs,
    # peripheral_bones, ribs, vertebrae} models without brain, skull and trachea). Trained with the current nnU-Net:
    # its plans name the reorienting reader (NibabelIOWithReorient, read with SimpleITKIOWithReorient here) and the
    # resampling it was trained with, so the entry needs nothing else.
    # Its weights are placed into the models directory by hand (local URL).
    "clin_ct_anatomy": {
        KEY_URL: f"{LOCAL_URL_PREFIX}clin_ct_anatomy_302_fold5_ras",
        KEY_FOLDER_NAME: "Dataset302_Enhance-PET-1_6k_v2_CTOnly",
    },
    "clin_ct_body": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_body_27112023.zip",
        KEY_FOLDER_NAME: "Dataset001_body",
        **RAS_CUBIC,
    },
    "clin_ct_lungs": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_lungs_24062023.zip",
        KEY_FOLDER_NAME: "Dataset333_HMS3dlungs",
        **RAS_CUBIC,
        KEY_ORIENTATION: "LAS",
    },
    "clin_ct_organs": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_organs_ras_07052025.zip",
        KEY_FOLDER_NAME: "Dataset123_Organs",
        **RAS_CUBIC,
    },
    "clin_ct_ribs": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_ribs_ras_08052025.zip",
        KEY_FOLDER_NAME: "Dataset444_Ribs",
        **RAS_CUBIC,
    },
    "clin_ct_muscles": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_muscles_ras_21052025.zip",
        KEY_FOLDER_NAME: "Dataset555_Muscles",
        **RAS_CUBIC,
    },
    "clin_ct_peripheral_bones": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_peripheral_bones_ras_07052025.zip",
        KEY_FOLDER_NAME: "Dataset666_Peripheral-Bones",
        **RAS_CUBIC,
    },
    "clin_ct_vertebrae": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_vertebrae_ras_21052025.zip",
        KEY_FOLDER_NAME: "Dataset111_Vertebrae",
        **RAS_CUBIC,
    },
    "clin_ct_cardiac": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_cardiac_ras_21052025.zip",
        KEY_FOLDER_NAME: "Dataset888_Cardiac",
        **RAS_CUBIC,
    },
    "clin_ct_digestive": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_digestive_06102024.zip",
        KEY_FOLDER_NAME: "Dataset999_Digestive",
        **RAS_CUBIC,
    },
    "clin_ct_ALPACA": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_ALPACA.zip",
        KEY_FOLDER_NAME: "Dataset080_Alpaca",
        **RAS_CUBIC,
        KEY_ORIENTATION: "LAS",
    },
    "clin_ct_PUMA": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_PUMA_1k_23052024.zip",
        KEY_FOLDER_NAME: "Dataset002_PUMA",
        **RAS_CUBIC,
        KEY_ORIENTATION: "LAS",
    },
    "clin_ct_PUMA4": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_PUMA4_06032024.zip",
        KEY_FOLDER_NAME: "Dataset003_PUMA4",
        **RAS_CUBIC,
    },
    "clin_ct_body_composition": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_body_composition_05092024.zip",
        KEY_FOLDER_NAME: "Dataset778_Body_composition",
        **RAS_CUBIC,
        # runs on the slab from L1 to L5 and keeps the slices of L3
        KEY_ROI: {"model": "clin_ct_fast_vertebrae", "crop_labels": (20, 21, 22, 23, 24), "keep_label": 22},
    },
    "clin_ct_fast_organs": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_organs_6_02092024.zip",
        KEY_FOLDER_NAME: "Dataset145_Fast_organs",
        **RAS_CUBIC,
    },
    "clin_ct_fast_vertebrae": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_vertebrae3_10092024.zip",
        KEY_FOLDER_NAME: "Dataset112_FastVertebrae",
        **RAS_CUBIC,
    },
    "clin_ct_fast_cardiac": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_cardiac3_10092024.zip",
        KEY_FOLDER_NAME: "Dataset890_FastCardiac",
        **RAS_CUBIC,
        KEY_ORIENTATION: "LAS",
    },
    "clin_ct_all_bones_v1": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_all_bones_25102023.zip",
        KEY_FOLDER_NAME: "Dataset600_Original_bones",
        **RAS_CUBIC,
    },
    "clin_pt_fdg_brain_v1": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_fdg_pt_brain_v1_17112023.zip",
        KEY_FOLDER_NAME: "Dataset100_Brain_v1",
        **RAS_CUBIC,
    },
    "clin_ct_fat_old": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_fat_31082023.zip",
        KEY_FOLDER_NAME: "Dataset777_Fat",
        **RAS_CUBIC,
    },
    "clin_ct_dental": {
        KEY_URL: "https://model.s.mdforge.com/Dataset112_DentalSegmentator_v100_moose.zip",
        KEY_FOLDER_NAME: "Dataset112_DentalSegmentator_v100",
        **RAS_CUBIC,
    },
    "clin_ct_face": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.7/clin_ct_face_ear_12052026.zip",
        KEY_FOLDER_NAME: "Dataset006_face_ear_v2",
        **RAS_CUBIC,
    },
    "clin_pt_fdg_face": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.5/clin_pt_face_12012026.zip",
        KEY_FOLDER_NAME: "Dataset183_PETface",
        **RAS_CUBIC,
    },
    "preclin_mr_all": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/preclin_mr_all_05122023.zip",
        KEY_FOLDER_NAME: "Dataset234_minimoose",
        **RAS_CUBIC,
    },
    "preclin_ct_legs": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/preclin_ct_legs_05122023.zip",
        KEY_FOLDER_NAME: "Dataset256_Preclin_leg_muscles",
        **RAS_CUBIC,
    },
    "clin_mr_FVM": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.2.0/clin_mr_FVM_30032026.zip",
        KEY_FOLDER_NAME: "Dataset501_FVM",
        **RAS_CUBIC,
    } 
}

AVAILABLE_MODELS = MODEL_METADATA.keys()


class Model:
    def __init__(self, model_identifier: str, output_manager: system.OutputManager, base_directory: str = None):
        if base_directory is None:
            base_directory = system.MODELS_DIRECTORY_PATH
        self.model_identifier = model_identifier
        self.folder_name = MODEL_METADATA[self.model_identifier][KEY_FOLDER_NAME]
        self.url = MODEL_METADATA[self.model_identifier][KEY_URL]
        self.base_directory = base_directory
        self.directory = os.path.join(self.base_directory, self.folder_name)

        self.__download(output_manager)
        self.configuration_folders = self.__get_configuration_folders(output_manager)
        self.configuration_directory = os.path.join(self.directory, self.configuration_folders[0])
        self.trainer, self.planner, self.resolution_configuration = self.__get_model_configuration()
        self.folds = self.__get_model_folds()

        self.dataset, self.plans = self.__get_model_data()
        metadata = MODEL_METADATA[self.model_identifier]
        plans_reader = self.plans["image_reader_writer"]
        self.reader = metadata.get(KEY_READER, SIMPLEITK_READER_FOR.get(plans_reader, plans_reader))
        if self.reader not in SUPPORTED_READERS:
            raise NotImplementedError(f"moosez cannot load images for {self.model_identifier}, read with {self.reader}; "
                                      f"it supports {', '.join(SUPPORTED_READERS)}")
        self.orientation = metadata.get(KEY_ORIENTATION, "RAS")
        self.input_resampling = metadata.get(KEY_INPUT_RESAMPLING)
        self.roi = metadata.get(KEY_ROI)
        self.imaging_type, self.modality, self.modality_subtype, self.modality_full, self.region = self.__get_model_identifier_segments()
        self.multilabel_prefix = f"{self.imaging_type}_{self.modality_full}_{self.region}_"

        self.organ_indices = self.__get_organ_indices()
        self.nr_training_data = self.__get_number_training_data()

    def get_expectation(self):
        expected_modalities = [self.modality_full]
        expected_prefixes = [self.modality_full + "_"]
        return expected_modalities, expected_prefixes

    def __get_configuration_folders(self, output_manager: system.OutputManager) -> List[str]:
        items = os.listdir(self.directory)
        folders = [item for item in items if not item.startswith(".") and item.count("__") == 2 and os.path.isdir(os.path.join(self.directory, item))]

        if len(folders) > 1:
            output_manager.console_update("Information: more than one configuration folder found. Utilizing information of the first one encountered.")

        if not folders:
            raise ValueError(f"No valid configuration folders found in {self.directory}")

        return folders

    def __get_model_configuration(self) -> Tuple[str, str, str]:
        model_configuration_folder = os.path.basename(self.configuration_directory)
        trainer, planner, resolution_configuration = model_configuration_folder.split("__")
        return trainer, planner, resolution_configuration

    def __get_model_folds(self) -> Tuple[str, ...]:
        folds_folders = [item for item in os.listdir(self.configuration_directory) if os.path.isdir(os.path.join(self.configuration_directory, item))]
        folds = tuple([fold.replace("fold_", "") for fold in folds_folders])
        return folds

    def __get_model_identifier_segments(self) -> Tuple[str, str, str | None, str, str]:
        segments = self.model_identifier.split('_')

        imaging_type = segments[0]
        modality = segments[1].upper()

        if modality not in ALLOWED_MODALITY_CONFIGURATIONS:
            raise ValueError(f"Invalid modality: {modality}")

        allowed_subtypes = ALLOWED_MODALITY_CONFIGURATIONS[modality]

        if None not in allowed_subtypes:
            modality_subtype = segments[2].upper()
            if modality_subtype not in allowed_subtypes:
                raise ValueError(f"Invalid modality subtype: {modality_subtype}")
            modality_full = f"{modality}_{modality_subtype}"
            region = '_'.join(segments[3:])
        else:
            known_subtypes = [s for s in allowed_subtypes if s is not None]
            if known_subtypes and segments[2].upper() in known_subtypes:
                modality_subtype = segments[2].upper()
                modality_full = f"{modality}_{modality_subtype}"
                region = '_'.join(segments[3:])
            else:
                modality_subtype = None
                modality_full = modality
                region = '_'.join(segments[2:])

        return imaging_type, modality, modality_subtype, modality_full, region

    def __get_model_data(self) -> Tuple[Dict, Dict]:
        dataset_json_path = os.path.join(self.configuration_directory, FILE_NAME_DATASET_JSON)
        plans_json_path = os.path.join(self.configuration_directory, FILE_NAME_PLANS_JSON)
        try:
            with open(dataset_json_path) as f:
                dataset = json.load(f)

            with open(plans_json_path) as f:
                plans = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load model data from {dataset_json_path} or {plans_json_path}: {e}")

        return dataset, plans

    def __download(self, output_manager: system.OutputManager):
        """
        Download and extract the model if it does not exist locally
        or if the existing folder has a different URL recorded.
        """
        # Weights placed by hand: use them as they are, never remove them
        if self.url.startswith(LOCAL_URL_PREFIX):
            if not os.path.isdir(self.directory):
                raise FileNotFoundError(f" The weights of '{self.model_identifier}' are not downloaded automatically. Install "
                                        f"its model archive with 'moosez -mi <archive>.zip', or place the model folder "
                                        f"at {self.directory}.")
            output_manager.log_update(f"    - A local instance of {self.model_identifier} has been detected.")
            return

        # If folder already exists, check if we should remove it
        if os.path.exists(self.directory):
            # Attempt to load the previously saved URL from version file
            version_file_path = os.path.join(self.directory, "model_version.json")
            old_url = None

            if os.path.exists(version_file_path):
                try:
                    with open(version_file_path, 'r') as vf:
                        version_data = json.load(vf)
                        old_url = version_data.get("url")
                except Exception:
                    pass  # If JSON is corrupted, we'll just treat it as mismatch

            # If the existing folder's URL doesn't match the new URL, remove folder
            if old_url != self.url:
                output_manager.console_update(f" Model version mismatch detected for '{self.model_identifier}'. Removing outdated model and downloading the latest model...")
                shutil.rmtree(self.directory, ignore_errors=True)
            else:
                # If the URL matches, we skip re-downloading
                output_manager.log_update(f"    - A local instance of {self.model_identifier} has been detected.")
                output_manager.console_update(f"{ANSI_GREEN} A local instance of {self.model_identifier} has been detected. {ANSI_RESET}")
                return

        # If folder doesn't exist or has been removed, proceed to download
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

        if not self.url:
            raise ValueError(f" No URL specified for model '{self.model_identifier}'.")

        output_manager.log_update(f"    - Downloading {self.model_identifier}")
        download_file_name = os.path.basename(self.url)
        download_file_path = os.path.join(self.base_directory, download_file_name)

        response = requests.get(self.url, stream=True)
        if response.status_code != 200:
            output_manager.log_update(f"    X Failed to download model from {self.url}")
            raise Exception(f" Failed to download model from {self.url}")

        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 10

        progress = output_manager.create_file_progress_bar()
        with progress:
            task = progress.add_task(f"[white] Downloading {self.model_identifier}...", total=total_size)
            with open(download_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=chunk_size)

        output_manager.log_update(f"    - {self.model_identifier} ({self.folder_name}) downloaded.")

        # Extract next to the other models, then move the model folder into place: some archives hold it inside
        # another folder (clin_mr_FVM)
        progress = output_manager.create_file_progress_bar()
        with progress, tempfile.TemporaryDirectory(dir=self.base_directory) as extraction_directory:
            with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
                total_size = sum(file.file_size for file in zip_ref.infolist())
                task = progress.add_task(f"[white] Extracting {self.model_identifier}...", total=total_size)
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extraction_directory)
                    progress.update(task, advance=file.file_size)
            os.remove(download_file_path)
            extracted = find_model_folders(extraction_directory, [self.folder_name])
            if not extracted:
                raise FileNotFoundError(f" The archive of '{self.model_identifier}' ({self.url}) does not hold the "
                                        f"folder {self.folder_name}.")
            shutil.move(extracted[self.folder_name], self.directory)

        output_manager.log_update(f"    - {self.model_identifier} extracted.")

        # Save a small version file with the URL
        version_file_path = os.path.join(self.directory, "model_version.json")
        with open(version_file_path, 'w') as vf:
            json.dump({"url": self.url}, vf)

        output_manager.log_update(f"    - {self.model_identifier} - setup complete.")
        output_manager.console_update(f"{ANSI_GREEN} {self.model_identifier} - setup complete. {ANSI_RESET}")

    def __get_organ_indices(self) -> Dict[int, str]:
        labels = self.dataset.get('labels', {})
        # label values are strings in moosez's own dataset.json files and integers in nnU-Net's
        return {int(value): key for key, value in labels.items() if int(value) != 0}

    def organ_indices_to_json(self, directory_path: str):
        organ_indices_mappings = {}
        for intensity, organ_name in self.organ_indices.items():
            organ_indices_mappings[intensity] = {"name": organ_name}
            if organ_name in SNOMED.moose_to_snomed:
                organ_indices_mappings[intensity]["SNOMED"] = SNOMED.moose_to_snomed[organ_name]

        file_path = os.path.join(directory_path, f"{self.multilabel_prefix}organ_indices.json")
        with open(file_path, "w") as organ_indices_json_file:
            json.dump({"organ_indices": organ_indices_mappings}, organ_indices_json_file, indent=4)

    def __get_number_training_data(self) -> str:
        nr_training_data = str(self.dataset.get('numTraining', "Not Available"))
        return nr_training_data

    def __str__(self):
        return self.model_identifier

    def __repr__(self):
        result = [
            f"Model Object of {self.model_identifier}",
            f" Folder Name: {self.folder_name}",
            f" URL: {self.url}",
            f" Reader: {self.reader} ({self.orientation})",
            f" Input Resampling: {self.input_resampling or 'as in the plans'}",
            f" Region of Interest: {self.roi}",
            f" Directory: {self.directory}",
            f" Configuration Directory: {self.configuration_directory}",
            f" Trainer: {self.trainer}",
            f" Planner: {self.planner}",
            f" Resolution Configuration: {self.resolution_configuration}",
            f" Imaging Type: {self.imaging_type}",
            f" Modality: {self.modality}",
            f" Modality Subtype: {self.modality_subtype}",
            f" Modality Full: {self.modality_full}",
            f" Region: {self.region}",
            f" Multilabel Prefix: {self.multilabel_prefix}",
            f" Organ Indices:",
        ]
        for index, organ in self.organ_indices.items():
            result.append(f"   {index}: {organ}")

        return "\n".join(result)

    @staticmethod
    def model_identifier_valid(model_identifier: str, output_manager: system.OutputManager) -> bool:
        if model_identifier not in MODEL_METADATA:
            output_manager.console_update("No valid model selected.")
            return False

        model_information = MODEL_METADATA[model_identifier]
        if KEY_URL not in model_information or KEY_FOLDER_NAME not in model_information:
            output_manager.console_update("One or more of the required keys url, folder_name, are missing.")
            return False

        if model_information[KEY_URL] == "" or model_information[KEY_FOLDER_NAME] == "":
            output_manager.console_update("One or more of the required keys url, folder_name, are not defined correctly.")
            return False

        return True


def with_roi_models(requested: Sequence[Model], output_manager: system.OutputManager) -> Dict[str, Model]:
    """
    The requested models and, recursively, the models whose label maps give their roi, by identifier. Each model is
    built (and downloaded if needed) once.
    """
    by_identifier = {model.model_identifier: model for model in requested}
    for model in requested:
        while model.roi is not None:
            model = by_identifier.get(model.roi['model']) or Model(model.roi['model'], output_manager)
            by_identifier[model.model_identifier] = model
    return by_identifier


def find_model_folders(directory: str, folder_names: Sequence[str]) -> Dict[str, str]:
    """The folders named in folder_names anywhere below directory, {name: path} (the first one found of each name)."""
    found = {}
    for root, folders, _ in os.walk(directory):
        for name in folder_names:
            if name in folders and name not in found:
                found[name] = os.path.join(root, name)
    return found


def install_model_archive(archive_path: str, output_manager: system.OutputManager, base_directory: str = None) -> List[str]:
    """
    moosez -mi: puts the model folder(s) of a zip archive into the models directory, wherever the archive holds them,
    for every registry model whose folder it holds. An existing folder of that model is replaced. Returns the
    identifiers of the models installed.
    """
    if base_directory is None:
        base_directory = system.MODELS_DIRECTORY_PATH
    os.makedirs(base_directory, exist_ok=True)
    identifiers = {metadata[KEY_FOLDER_NAME]: identifier for identifier, metadata in MODEL_METADATA.items()}
    installed = []
    with tempfile.TemporaryDirectory(dir=base_directory) as extraction_directory:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_directory)
        for folder_name, path in find_model_folders(extraction_directory, list(identifiers)).items():
            identifier, directory = identifiers[folder_name], os.path.join(base_directory, folder_name)
            if os.path.exists(directory):
                shutil.rmtree(directory)
            shutil.move(path, directory)
            url = MODEL_METADATA[identifier][KEY_URL]
            if not url.startswith(LOCAL_URL_PREFIX):        # so that moosez takes it for the registry's version
                with open(os.path.join(directory, "model_version.json"), 'w') as version_file:
                    json.dump({"url": url}, version_file)
            output_manager.log_update(f"    - {identifier} installed from {archive_path} into {directory}")
            installed.append(identifier)
    return installed
