import os
import json
import zipfile
import requests
import shutil
from typing import Union, Tuple, List, Dict
from moosez import system
from moosez.constants import (KEY_FOLDER_NAME, KEY_URL, KEY_LIMIT_FOV, DEFAULT_SPACING, FILE_NAME_DATASET_JSON,
                              FILE_NAME_PLANS_JSON, ANSI_GREEN, ANSI_RESET)
from moosez.mappings import SNOMED


MODEL_METADATA = {
    "clin_ct_lungs": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_lungs_24062023.zip",
        KEY_FOLDER_NAME: "Dataset333_HMS3dlungs",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_organs": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_organs_25022025.zip",
        KEY_FOLDER_NAME: "Dataset123_Organs",
        KEY_LIMIT_FOV: None
    },
    "preclin_mr_all": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/preclin_mr_all_05122023.zip",
        KEY_FOLDER_NAME: "Dataset234_minimoose",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_body": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_body_27112023.zip",
        KEY_FOLDER_NAME: "Dataset001_body",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_ribs": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_ribs_11082024.zip",
        KEY_FOLDER_NAME: "Dataset444_Ribs",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_muscles": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_muscles_09082024.zip",
        KEY_FOLDER_NAME: "Dataset555_Muscles",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_peripheral_bones": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_peripheral_bones_22022025.zip",
        KEY_FOLDER_NAME: "Dataset666_Peripheral-Bones",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_vertebrae": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_vertebrae_11082024.zip",
        KEY_FOLDER_NAME: "Dataset111_Vertebrae",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_cardiac": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_cardiac_09082024.zip",
        KEY_FOLDER_NAME: "Dataset888_Cardiac",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_digestive": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_digestive_06102024.zip",
        KEY_FOLDER_NAME: "Dataset999_Digestive",
        KEY_LIMIT_FOV: None
    },
    "preclin_ct_legs": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/preclin_ct_legs_05122023.zip",
        KEY_FOLDER_NAME: "Dataset256_Preclin_leg_muscles",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_all_bones_v1": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_all_bones_25102023.zip",
        KEY_FOLDER_NAME: "Dataset600_Original_bones",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_PUMA": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_PUMA_1k_23052024.zip",
        KEY_FOLDER_NAME: "Dataset002_PUMA",
        KEY_LIMIT_FOV: None
    },
    "clin_pt_fdg_brain_v1": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_fdg_pt_brain_v1_17112023.zip",
        KEY_FOLDER_NAME: "Dataset100_Brain_v1",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_ALPACA": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_ALPACA.zip",
        KEY_FOLDER_NAME: "Dataset080_Alpaca",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_PUMA4": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_PUMA4_06032024.zip",
        KEY_FOLDER_NAME: "Dataset003_PUMA4",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_fast_organs": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_organs_6_02092024.zip",
        KEY_FOLDER_NAME: "Dataset145_Fast_organs",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_body_composition": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_body_composition_05092024.zip",
        KEY_FOLDER_NAME: "Dataset778_Body_composition",
        KEY_LIMIT_FOV: {
            "model_to_crop_from": "clin_ct_fast_vertebrae",
            "inference_fov_intensities": [20, 24],
            "label_intensity_to_crop_from": 22,
            "largest_component_only": True
        }
    },
    "clin_ct_fast_vertebrae": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_vertebrae3_10092024.zip",
        KEY_FOLDER_NAME: "Dataset112_FastVertebrae",
        KEY_LIMIT_FOV: None
        },
    "clin_ct_fast_cardiac": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_cardiac3_10092024.zip",
        KEY_FOLDER_NAME: "Dataset890_FastCardiac",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_fat_old": {
        KEY_URL: "https://enhance-pet.s3.eu-central-1.amazonaws.com/moose/clin_ct_fat_31082023.zip",
        KEY_FOLDER_NAME: "Dataset777_Fat",
        KEY_LIMIT_FOV: None
    },
    "clin_ct_dental": {
        KEY_URL: "https://model.s.mdforge.com/Dataset112_DentalSegmentator_v100_moose.zip",
        KEY_FOLDER_NAME: "Dataset112_DentalSegmentator_v100",
        KEY_LIMIT_FOV: None
    }
}

AVAILABLE_MODELS = MODEL_METADATA.keys()


class Model:
    def __init__(self, model_identifier: str, output_manager: system.OutputManager, override_directory: str = None):
        self.model_identifier = model_identifier
        self.folder_name = MODEL_METADATA[self.model_identifier][KEY_FOLDER_NAME]
        self.url = MODEL_METADATA[self.model_identifier][KEY_URL]
        self.limit_fov = MODEL_METADATA[self.model_identifier][KEY_LIMIT_FOV]

        base_model_dir = os.path.abspath(override_directory) if override_directory else system.MODELS_DIRECTORY_PATH
        self.directory = os.path.join(base_model_dir, self.folder_name)

        self.__download(output_manager)
        self.configuration_folders = self.__get_configuration_folders(output_manager)
        self.configuration_directory = os.path.join(self.directory, self.configuration_folders[0])
        self.trainer, self.planner, self.resolution_configuration = self.__get_model_configuration()

        self.dataset, self.plans = self.__get_model_data()
        self.voxel_spacing = tuple(self.plans.get('configurations').get(self.resolution_configuration).get('spacing', DEFAULT_SPACING))
        self.imaging_type, self.modality, self.region = self.__get_model_identifier_segments()
        self.multilabel_prefix = f"{self.imaging_type}_{self.modality}_{self.region}_"

        self.organ_indices = self.__get_organ_indices()
        self.nr_training_data = self.__get_number_training_data()

    def get_expectation(self):
        if self.modality == 'FDG-PET-CT':
            expected_modalities = ['FDG-PET', 'CT']
        else:
            expected_modalities = [self.modality]
        expected_prefixes = [m.replace('-', '_') + "_" for m in expected_modalities]

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

    def __get_model_identifier_segments(self) -> Tuple[str, str, str]:
        segments = self.model_identifier.split('_')

        imaging_type = segments[0]
        if segments[1] == 'pt':
            modality = f'{segments[1]}_{segments[2]}'.upper()
            region = '_'.join(segments[3:])
        else:
            modality = segments[1].upper()
            region = '_'.join(segments[2:])

        return imaging_type, modality, region

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
                output_manager.console_update(
                    f" Model version mismatch detected for '{self.model_identifier}'. Removing outdated model and downloading the latest model..."

                )
                shutil.rmtree(self.directory, ignore_errors=True)
            else:
                # If the URL matches, we skip re-downloading
                output_manager.log_update(
                    f"    - A local instance of {self.model_identifier} has been detected."
                )
                output_manager.console_update(
                    f"{ANSI_GREEN} A local instance of {self.model_identifier} has been detected. {ANSI_RESET}"
                )
                return

        # If folder doesn't exist or has been removed, proceed to download
        if not os.path.exists(system.MODELS_DIRECTORY_PATH):
            os.makedirs(system.MODELS_DIRECTORY_PATH)

        if not self.url:
            raise ValueError(f" No URL specified for model '{self.model_identifier}'.")

        output_manager.log_update(f"    - Downloading {self.model_identifier}")
        download_file_name = os.path.basename(self.url)
        download_file_path = os.path.join(system.MODELS_DIRECTORY_PATH, download_file_name)

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

        output_manager.log_update(
            f"    - {self.model_identifier} ({self.folder_name}) downloaded."
        )

        # Extract
        progress = output_manager.create_file_progress_bar()
        with progress:
            with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
                total_size = sum(file.file_size for file in zip_ref.infolist())
                task = progress.add_task(f"[white] Extracting {self.model_identifier}...", total=total_size)
                for file in zip_ref.infolist():
                    extract_root = os.path.dirname(self.directory)
                    os.makedirs(extract_root, exist_ok=True)
                    zip_ref.extract(file, extract_root)
                    progress.update(task, advance=file.file_size)

        output_manager.log_update(f"    - {self.model_identifier} extracted.")
        os.remove(download_file_path)

        # Save a small version file with the URL
        os.makedirs(self.directory, exist_ok=True)
        version_file_path = os.path.join(self.directory, "model_version.json")
        with open(version_file_path, 'w') as vf:
            json.dump({"url": self.url}, vf)

        output_manager.log_update(f"    - {self.model_identifier} - setup complete.")
        output_manager.console_update(
            f"{ANSI_GREEN} {self.model_identifier} - setup complete. {ANSI_RESET}"
        )

    def __get_organ_indices(self) -> Dict[int, str]:
        labels = self.dataset.get('labels', {})
        return {int(value): key for key, value in labels.items() if value != "0"}

    def organ_indices_to_json(self, directory_path: str):
        organ_indices_mappings = {}
        for intensity, organ_name in self.organ_indices.items():
            organ_indices_mappings[intensity] = {"name": organ_name,
                                                 "SNOMED": SNOMED.moose_to_snomed[organ_name]}
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
            f" Directory: {self.directory}",
            f" Configuration Directory: {self.configuration_directory}",
            f" Trainer: {self.trainer}",
            f" Planner: {self.planner}",
            f" Resolution Configuration: {self.resolution_configuration}",
            f" Voxel Spacing: {self.voxel_spacing}",
            f" Imaging Type: {self.imaging_type}",
            f" Modality: {self.modality}",
            f" Region: {self.region}",
            f" Multilabel Prefix: {self.multilabel_prefix}",
            f" Organ Indices:",
        ]
        for index, organ in self.organ_indices.items():
            result.append(f"   {index}: {organ}")

        if isinstance(self.limit_fov, dict):
            result.append(f" Limit FOV:")
            for key, value in self.limit_fov.items():
                result.append(f"   {key}: {value}")
        else:
            result.append(f" Limit FOV: {self.limit_fov}")

        return "\n".join(result)

    @staticmethod
    def model_identifier_valid(model_identifier: str, output_manager: system.OutputManager) -> bool:
        if model_identifier not in MODEL_METADATA:
            output_manager.console_update("No valid model selected.")
            return False

        model_information = MODEL_METADATA[model_identifier]
        if KEY_URL not in model_information or KEY_FOLDER_NAME not in model_information or KEY_LIMIT_FOV not in model_information:
            output_manager.console_update("One or more of the required keys url, folder_name, limit_fov are missing.")
            return False

        if model_information[KEY_URL] == "" or model_information[KEY_FOLDER_NAME] == "" or (model_information[KEY_LIMIT_FOV] is not None and not isinstance(model_information[KEY_LIMIT_FOV], dict)):
            output_manager.console_update("One or more of the required keys url, folder_name, limit_fov are not defined correctly.")
            return False

        return True


class ModelWorkflow:
    def __init__(self, model_identifier: str, output_manager: system.OutputManager):
        self.workflow: List[Model] = []
        self.__construct_workflow(model_identifier, output_manager)
        if self.workflow:
            self.initial_desired_spacing = self.workflow[0].voxel_spacing
            self.target_model = self.workflow[-1]

    def __construct_workflow(self, model_identifier: str, output_manager: system.OutputManager):
        model = Model(model_identifier, output_manager)
        if model.limit_fov and isinstance(model.limit_fov, dict) and 'model_to_crop_from' in model.limit_fov:
            self.__construct_workflow(model.limit_fov["model_to_crop_from"], output_manager)
        self.workflow.append(model)

    def __len__(self) -> len:
        return len(self.workflow)

    def __getitem__(self, index) -> Model:
        return self.workflow[index]

    def __iter__(self):
        return iter(self.workflow)

    def __str__(self) -> str:
        return " -> ".join([model.model_identifier for model in self.workflow])


def construct_model_routine(model_identifiers: Union[str, List[str]], output_manager: system.OutputManager) -> Dict[tuple, List[ModelWorkflow]]:
    if isinstance(model_identifiers, str):
        model_identifiers = [model_identifiers]

    model_routine: Dict = {}
    output_manager.log_update(' SETTING UP MODEL WORKFLOWS:')
    for model_identifier in model_identifiers:
        output_manager.log_update(' - Model name: ' + model_identifier)
        model_workflow = ModelWorkflow(model_identifier, output_manager)

        if model_workflow.initial_desired_spacing in model_routine:
            model_routine[model_workflow.initial_desired_spacing].append(model_workflow)
        else:
            model_routine[model_workflow.initial_desired_spacing] = [model_workflow]

    return model_routine
