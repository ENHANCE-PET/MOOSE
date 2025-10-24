#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import torch
import os
import emoji
import pyfiglet
import importlib.metadata
from halo import Halo
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from rich.console import Console, RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, FileSizeColumn, TransferSpeedColumn, TimeRemainingColumn
from typing import Union, Tuple, List
from moosez.constants import VERSION, ANSI_VIOLET, ANSI_RESET


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

        self.nnunet_log_filename = os.path.join(log_file_directory, f'moosez-v{VERSION}_nnUNet_{timestamp}.log')

        self.logger = logging.getLogger(f'moosez-v{VERSION}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            log_format = '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            formatter = logging.Formatter(log_format)

            log_filename = os.path.join(log_file_directory, f'moosez-v{VERSION}_{timestamp}.log')
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
        version = importlib.metadata.version("moosez")
        self.console_update(' ')
        result = ANSI_VIOLET + pyfiglet.figlet_format(f" MOOSE {version}", font="smslant").rstrip() + ANSI_RESET
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

def check_device(output_manager: OutputManager = OutputManager(False, False)) -> Tuple[str, Union[int, None]]:
    """
    This function checks the available device for running predictions, considering CUDA and MPS (for Apple Silicon).

    Returns:
        str: The device to run predictions on, either "cpu", "cuda", or "mps".
    """
    # Check for CUDA
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        output_manager.console_update(f" CUDA is available with {device_count} GPU(s). Predictions will be run on GPU.")
        return "cuda", device_count
    # Check for MPS (Apple Silicon) Here for the future but not compatible right now
    elif torch.backends.mps.is_available():
        output_manager.console_update(" Apple MPS backend is available. Predictions will be run on Apple Silicon GPU.")
        return "mps", None
    elif not torch.backends.mps.is_built():
        output_manager.console_update(" MPS not available because the current PyTorch install was not built with MPS enabled.")
        return "cpu", None
    else:
        output_manager.console_update(" CUDA/MPS not available. Predictions will be run on CPU.")
        return "cpu", None


os.environ["nnUNet_raw"] = ""
os.environ["nnUNet_preprocessed"] = ""
os.environ["nnUNet_results"] = ""
MOOSE_ROOT_PATH: str = os.path.dirname(os.path.abspath(__file__))
MODELS_DIRECTORY_PATH: str = os.path.join(MOOSE_ROOT_PATH, 'models', 'nnunet_trained_models')
