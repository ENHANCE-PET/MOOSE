#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import torch
import os
import emoji
import pyfiglet
import platform
import json
import importlib.metadata
from halo import Halo
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from rich.console import Console, RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, FileSizeColumn, TransferSpeedColumn, TimeRemainingColumn
from typing import Union, Tuple, List, Dict, Optional
from moosez.constants import ANSI_VIOLET, ANSI_RESET


# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module contains the urls and filenames of the models and binaries that are required for the moosez.
#
# Usage:
# The variables in this module can be imported and used in other modules within the moosez to download the necessary
# binaries and models for the moosez.
#
# ----------------------------------------------------------------------------------------------------------------------


class OutputManager:
    def __init__(self, verbose_console: bool, verbose_log: bool):
        self.verbose_console = verbose_console
        self.verbose_log = verbose_log

        self.console = Console(quiet=not self.verbose_console)
        self.spinner = Halo(spinner='dots', enabled=self.verbose_console)
        self.spinner_running = False

        self.logger = None
        self.nnunet_log_filename = os.devnull

        # On Windows, the console encoding may not be UTF-8, which can cause issues with certain characters
        self.sanitize_console_encoding = self.console.encoding.lower() not in ['utf-8', 'utf8']

    def create_file_progress_bar(self):
        progress_bar = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=None),
                                "[progress.percentage]{task.percentage:>3.0f}%", "•", FileSizeColumn(),
                                TransferSpeedColumn(), TimeRemainingColumn(), console=self.console, expand=True)
        return progress_bar

    def create_progress_bar(self):
        progress_bar = Progress(console=self.console)
        return progress_bar

    def create_table(self, header: List[str], styles: Union[List[str], None] = None) -> Table:
        table = Table()
        if styles is None:
            styles = [None] * len(header)
        for header, style in zip(header, styles):
            table.add_column(header, style = style)
        return table

    def configure_logging(self, log_file_directory: Union[str, None]):
        if not self.verbose_log or self.logger:
            return

        if log_file_directory is None:
            log_file_directory = os.getcwd()

        timestamp = datetime.now().strftime('%H-%M-%d-%m-%Y')

        self.nnunet_log_filename = os.path.join(log_file_directory, f'moosez-v{MOOSE_VERSION}_nnUNet_{timestamp}.log')

        self.logger = logging.getLogger(f'moosez-v{MOOSE_VERSION}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            log_format = '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            formatter = logging.Formatter(log_format)

            log_filename = os.path.join(log_file_directory, f'moosez-v{MOOSE_VERSION}_{timestamp}.log')
            file_handler = logging.FileHandler(log_filename, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def log_update(self, text: str):
        if self.verbose_log and self.logger:
            self.logger.info(text)

    def console_update(self, text: Union[str, RenderableType]):
        if isinstance(text, str):
            text = Text.from_ansi(text)

        if self.sanitize_console_encoding and isinstance(text, Text):
            # Ensure the text can be represented in the console by robust encoding and decoding (ignoring invalid characters)
            text = Text(text.plain.encode(self.console.encoding, "ignore").decode(self.console.encoding))

        self.console.print(text)

    def spinner_update(self, text: str):
        if self.spinner.enabled:
            if not self.spinner_running:
                self.spinner.start()
                self.spinner_running = True
            self.spinner.text = text

    def spinner_stop(self):
        if self.spinner.enabled and self.spinner_running:
            self.spinner.stop()
            self.spinner_running = False

    def spinner_start(self, text: str):
        if self.spinner.enabled and not self.spinner_running:
            self.spinner.start(text)
            self.spinner_running = True

    def spinner_succeed(self, text: str):
        if self.spinner.enabled:
            self.spinner.succeed(text)
            self.spinner_running = False

    def spinner_warn(self, text: str):
        if self.spinner.enabled:
            self.spinner.warn(text)
            self.spinner_running = False

    @contextmanager
    def manage_nnUNet_output(self):
        target_path = self.nnunet_log_filename if self.verbose_log else os.devnull
        mode = "a" if self.verbose_log else 'w'
        with open(target_path, mode) as target, redirect_stdout(target), redirect_stderr(target):
            yield

    def display_logo(self):
        """
        Display MOOSE logo

        This function displays the MOOSE logo using the pyfiglet library and ANSI color codes.

        :return: None
        """
        self.console_update(' ')
        result = ANSI_VIOLET + pyfiglet.figlet_format(f" MOOSE {MOOSE_VERSION}", font="smslant").rstrip() + ANSI_RESET
        text = ANSI_VIOLET + " A part of the ENHANCE community. Join us at www.enhance.pet to build the future of" \
                             " PET imaging together." + ANSI_RESET
        self.console_update(result)
        self.console_update(text)
        self.console_update(' ')

    def display_authors(self):
        """
        Display manuscript citation

        This function displays authors for the MOOSE project.

        :return: None
        """
        self.console_update(f'{ANSI_VIOLET} {emoji.emojize(":desktop_computer:")}  AUTHORS:{ANSI_RESET}')
        self.console_update(" ")
        self.console_update(" The Three Moose-keteers 🤺: Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer | Manuel Pires")
        self.console_update(" ")

    def display_doi(self):
        """
        Display manuscript citation

        This function displays the manuscript citation for the MOOSE project.

        :return: None
        """
        self.console_update(f'{ANSI_VIOLET} {emoji.emojize(":scroll:")} CITATION:{ANSI_RESET}')
        self.console_update(" ")
        self.console_update(" Fully Automated, Semantic Segmentation of Whole-Body [18F]-FDG PET/CT Images Based on Data-Centric Artificial Intelligence")
        self.console_update(" 10.2967/jnumed.122.264063")
        self.console_update(" ")
        self.console_update(" Copyright 2022, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")

    def display_docker_usage(self, docker_bind: str, example_model: str):
        docker_example = f"""\
        docker run --gpus all --ipc=host \\
        -v /path/to/data:/app/data \\
        -v {docker_bind}:/usr/local/models \\
        lalithshiyam/moosez \\
        -d /app/data -m {example_model}
        """
        docker_msg = Text()
        docker_msg.append("🐳 Some useful information...\n", style="bold violet")
        docker_msg.append("\n📂 Downloaded models are stored here:\n", style="bold")
        docker_msg.append(f"  {docker_bind}/nnunet_trained_models\n\n", style="orange3")
        docker_msg.append("📦 Bind this folder into the container at:\n", style="bold")
        docker_msg.append("  /usr/local/models\n\n", style="orange3")
        docker_msg.append("🚀 Example Docker Run:\n", style="bold")
        docker_msg.append(docker_example, style="green")

        self.console.print(Panel(docker_msg, title="🐳 INFO FOR DOCKER USERS", border_style="violet", padding=(1, 2)))


def check_device(output_manager: Optional[OutputManager] = None) -> Tuple[str, Union[int, None]]:
    """
    This function checks the available device for running predictions, considering CUDA, MPS (for Apple Silicon) or CPU as fallback.
    """
    if output_manager is None:
        output_manager = OutputManager(False, False)

    accelerator_information = get_accelerator_information()
    accelerator = accelerator_information["accelerator"]

    if accelerator == "cuda":
        output_manager.console_update(f" CUDA is available with {accelerator_information['device_count']} GPU(s). Predictions will be run on GPU.")
        return "cuda", accelerator_information["device_count"]

    if accelerator == "mps":
        output_manager.console_update(" Apple MPS backend is available. Predictions will be run on Apple Silicon GPU.")
        return "mps", None

    output_manager.console_update(" CUDA/MPS not available. Predictions will be run on CPU.")
    return "cpu", None


def get_accelerator_information() -> Dict:
    if torch.cuda.is_available():
        return {"accelerator": "cuda",
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_name": ", ".join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])}
    elif torch.backends.mps.is_available():
        return {"accelerator": "mps"}
    else:
        return {"accelerator": "cpu"}


def get_system_information() -> Dict:
    metadata = {"tool_name": "moosez",
                "tool_version": MOOSE_VERSION,
                "timestamp_start": datetime.now().strftime('%d.%m.%Y %H:%M'),
                "python_version": platform.python_version(),
                "os": platform.system(),"platform": platform.platform(),
                "architecture": platform.machine()}
    metadata.update(get_accelerator_information())

    return metadata


def write_information_json(metadata: Dict, metadata_file_directory: str) -> None:
    metadata["timestamp_stop"] = datetime.now().strftime('%d.%m.%Y %H:%M')

    os.makedirs(metadata_file_directory, exist_ok=True)
    metadata_file_path = os.path.join(metadata_file_directory, "MOOSE_information.json")
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4)


os.environ["nnUNet_raw"] = ""
os.environ["nnUNet_preprocessed"] = ""
os.environ["nnUNet_results"] = ""
try:
    MOOSE_VERSION = importlib.metadata.version("moosez")
except importlib.metadata.PackageNotFoundError:
    MOOSE_VERSION = "0.0.0"
MOOSE_ROOT_PATH: str = os.path.dirname(os.path.abspath(__file__))
MODELS_DIRECTORY_PATH: str = os.path.join(MOOSE_ROOT_PATH, 'models', 'nnunet_trained_models')
