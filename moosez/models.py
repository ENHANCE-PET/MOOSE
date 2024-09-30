import os
import json
import zipfile
import requests
from moosez.constants import (KEY_FOLDER_NAME, KEY_URL, KEY_LIMIT_FOV,
                              KEY_DESCRIPTION, KEY_DESCRIPTION_IMAGING, KEY_DESCRIPTION_MODALITY, KEY_DESCRIPTION_TEXT,
                              DEFAULT_SPACING, FILE_NAME_DATASET_JSON, FILE_NAME_PLANS_JSON, ANSI_GREEN, ANSI_RESET)
from moosez import resources


class Model:
    def __init__(self, model_identifier: str, output_manager: resources.OutputManager):
        self.model_identifier = model_identifier
        self.folder_name = resources.MODELS[self.model_identifier][KEY_FOLDER_NAME]
        self.url = resources.MODELS[self.model_identifier][KEY_URL]
        self.limit_fov = resources.MODELS[self.model_identifier][KEY_LIMIT_FOV]
        self.directory = os.path.join(resources.MODELS_DIRECTORY_PATH, self.folder_name)

        self.__download()
        self.configuration_folders = self.__get_configuration_folders()
        self.configuration_directory = os.path.join(self.directory, self.configuration_folders[0])
        self.trainer, self.planner, self.resolution_configuration = self.__get_model_configuration()

        self.dataset, self.plans = self.__get_model_data()
        self.voxel_spacing = tuple(self.plans.get('configurations').get(self.resolution_configuration).get('spacing', DEFAULT_SPACING))
        self.study_type, self.modality, self.region = self.__get_model_identifier_segments()
        self.multilabel_prefix = f"{self.study_type}_{self.modality}_{self.region}_"
        self.description = self.__create_description()
        self.organ_indices = self.__get_organ_indices()

        self.output_manager = output_manager

    def __create_description(self) -> dict:
        description = resources.MODELS[self.model_identifier][KEY_DESCRIPTION]
        description[KEY_DESCRIPTION_MODALITY] = self.modality
        if self.study_type == "clin":
            imaging = "Clinical"
        elif self.study_type == "preclin":
            imaging = "Pre-clinical"
        else:
            imaging = "unknown"
        description[KEY_DESCRIPTION_IMAGING] = imaging
        return description

    def get_expectation(self):
        if self.modality == 'FDG-PET-CT':
            expected_modalities = ['FDG-PET', 'CT']
        else:
            expected_modalities = [self.modality]
        expected_prefixes = [m.replace('-', '_') + "_" for m in expected_modalities]

        self.output_manager.console_update(f" Tissue of interest: {self.description[KEY_DESCRIPTION_TEXT]:<42} | Imaging: {self.description[KEY_DESCRIPTION_IMAGING]:<12} | Modality: {self.modality:<6}")
        return expected_modalities, expected_prefixes

    def __get_configuration_folders(self) -> list[str]:
        items = os.listdir(self.directory)
        folders = [item for item in items if not item.startswith(".") and item.count("__") == 2 and os.path.isdir(os.path.join(self.directory, item))]

        if len(folders) > 1:
            self.output_manager.console_update("Information: more than one configuration folder found. Utilizing information of the first one encountered.")

        if not folders:
            raise ValueError(f"No valid configuration folders found in {self.directory}")

        return folders

    def __get_model_configuration(self) -> tuple[str, str, str]:
        model_configuration_folder = os.path.basename(self.configuration_directory)
        trainer, planner, resolution_configuration = model_configuration_folder.split("__")
        return trainer, planner, resolution_configuration

    def __get_model_identifier_segments(self) -> tuple[str, str, str]:
        segments = self.model_identifier.split('_')

        study_type = segments[0]
        if segments[1] == 'pt':
            modality = f'{segments[1]}_{segments[2]}'.upper()
            region = '_'.join(segments[3:])
        else:
            modality = segments[1].upper()
            region = '_'.join(segments[2:])

        return study_type, modality, region

    def __get_model_data(self) -> tuple[dict, dict]:
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

    def __download(self):
        if os.path.exists(self.directory):
            self.output_manager.log_update(f"    - A local instance of {self.model_identifier} has been detected.")
            self.output_manager.console_update(f"{ANSI_GREEN} A local instance of {self.model_identifier} has been detected. {ANSI_RESET}")
            return

        self.output_manager.log_update(f"    - Downloading {self.model_identifier}")
        if not os.path.exists(resources.MODELS_DIRECTORY_PATH):
            os.makedirs(resources.MODELS_DIRECTORY_PATH)

        download_file_name = os.path.basename(self.url)
        download_file_path = os.path.join(resources.MODELS_DIRECTORY_PATH, download_file_name)

        response = requests.get(self.url, stream=True)
        if response.status_code != 200:
            self.output_manager.log_update(f"    X Failed to download model from {self.url}")
            raise Exception(f"Failed to download model from {self.url}")
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 10

        progress = self.output_manager.create_progress_bar()
        with progress:
            task = progress.add_task(f"[white] Downloading {self.model_identifier}...", total=total_size)
            with open(download_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=chunk_size)
        self.output_manager.log_update(f"    - {self.model_identifier} ({self.folder_name} downloaded.")

        progress = self.output_manager.create_progress_bar()
        with progress:
            with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
                total_size = sum((file.file_size for file in zip_ref.infolist()))
                task = progress.add_task(f"[white] Extracting {self.model_identifier}...", total=total_size)
                for file in zip_ref.infolist():
                    zip_ref.extract(file, resources.MODELS_DIRECTORY_PATH)
                    progress.update(task, advance=file.file_size)
        self.output_manager.log_update(f"    - {self.model_identifier} extracted.")

        os.remove(download_file_path)
        self.output_manager.log_update(f"    - {self.model_identifier} - setup complete.")
        self.output_manager.console_update(f"{ANSI_GREEN} {self.model_identifier} - setup complete. {ANSI_RESET}")

    def __get_organ_indices(self) -> dict[int, str]:
        labels = self.dataset.get('labels', {})
        return {int(value): key for key, value in labels.items() if value != "0"}

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
            f" Study Type: {self.study_type}",
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


def model_entry_valid(model_identifier: str) -> bool:
    if model_identifier not in resources.MODELS:
        print("No valid model selected.")
        return False

    model_information = resources.MODELS[model_identifier]
    if KEY_URL not in model_information or KEY_FOLDER_NAME not in model_information or KEY_LIMIT_FOV not in model_information:
        print("One or more of the required keys url, folder_name, limit_fov are missing.")
        return False

    if model_information[KEY_URL] == "" or model_information[KEY_FOLDER_NAME] == "" or (model_information[KEY_LIMIT_FOV] is not None and not isinstance(model_information[KEY_LIMIT_FOV], dict)):
        print("One or more of the required keys url, folder_name, limit_fov are not defined correctly.")
        return False

    return True


class ModelWorkflow:
    def __init__(self, model_identifier: str, output_manager: resources.OutputManager):
        self.workflow: list[Model] = []
        self.__construct_workflow(model_identifier, output_manager)
        if self.workflow:
            self.initial_desired_spacing = self.workflow[0].voxel_spacing
            self.target_model = self.workflow[-1]

    def __construct_workflow(self, model_identifier: str, output_manager: resources.OutputManager):
        model = Model(model_identifier, output_manager)
        if model.limit_fov and 'model_to_crop_from' in model.limit_fov:
            self.__construct_workflow(model.limit_fov["model_to_crop_from"], output_manager)
        self.workflow.append(model)

    def get_expectations(self):
        required_modalities = []
        required_prefixes = []

        for model in self.workflow:
            modalities, prefixes = model.get_expectation()
            required_modalities = required_modalities + modalities
            required_prefixes = required_prefixes + prefixes

        required_modalities = list(set(required_modalities))
        required_prefixes = list(set(required_prefixes))

        return required_modalities, required_prefixes

    def __len__(self) -> len:
        return len(self.workflow)

    def __getitem__(self, index) -> Model:
        return self.workflow[index]

    def __iter__(self):
        return iter(self.workflow)

    def __str__(self) -> str:
        return " -> ".join([model.model_identifier for model in self.workflow])


def construct_model_routine(model_identifiers: str | list[str], output_manager: resources.OutputManager) -> dict[tuple, list[ModelWorkflow]]:
    if isinstance(model_identifiers, str):
        model_identifiers = [model_identifiers]

    model_routine: dict = {}
    output_manager.log_update(' SETTING UP MODEL WORKFLOWS:')
    for model_identifier in model_identifiers:
        output_manager.log_update(' - Model name: ' + model_identifier)
        model_workflow = ModelWorkflow(model_identifier, output_manager)

        if model_workflow.initial_desired_spacing in model_routine:
            model_routine[model_workflow.initial_desired_spacing].append(model_workflow)
        else:
            model_routine[model_workflow.initial_desired_spacing] = [model_workflow]

    return model_routine
