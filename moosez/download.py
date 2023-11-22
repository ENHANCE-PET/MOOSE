#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import zipfile

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module downloads the necessary binaries and models for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to download the necessary
# binaries and models for the moosez.
#
# ----------------------------------------------------------------------------------------------------------------------
import requests
from moosez import constants
from moosez import resources
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, FileSizeColumn, TransferSpeedColumn, TimeRemainingColumn


def binary(system_info, url):
    """
    Downloads the binary for the current system.

    :param system_info: A dictionary containing the system information.
    :type system_info: dict
    :param url: The url to download the binary from.
    :type url: str
    """
    binary_name = "{}_{}_{}".format(system_info["os_type"], system_info["cpu_architecture"], system_info["cpu_brand"])
    print("Binary to download: " + binary_name)
    response = requests.get(url + binary_name)

    with open(binary_name, "wb") as f:
        f.write(response.content)


def model(model_name, model_path):
    """
    Downloads the model for the current system.

    :param model_name: The name of the model to download.
    :type model_name: str
    :param model_path: The path to store the model.
    :type model_path: str
    """
    model_info = resources.MODELS[model_name]
    url = model_info["url"]
    filename = os.path.join(model_path, model_info["filename"])
    directory = os.path.join(model_path, model_info["directory"])

    if not os.path.exists(directory):
        logging.info(f" Downloading {directory}")

        # Show progress using rich
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 10

        console = Console()
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            FileSizeColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        )

        with progress:
            task = progress.add_task(f"[white] Downloading {model_name}...", total=total_size)
            for chunk in response.iter_content(chunk_size=chunk_size):
                open(filename, "ab").write(chunk)
                progress.update(task, advance=chunk_size)

        # Unzip the model
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            FileSizeColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        )

        with progress:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                total_size = sum((file.file_size for file in zip_ref.infolist()))
                task = progress.add_task(f"[white] Extracting {model_name}...", total=total_size)
                parent_directory = os.path.dirname(directory)
                for file in zip_ref.infolist():
                    zip_ref.extract(file, parent_directory)
                    extracted_size = file.file_size
                    progress.update(task, advance=extracted_size)

        logging.info(f" {os.path.basename(directory)} extracted.")

        # Delete the zip file
        os.remove(filename)
        print(f"{constants.ANSI_GREEN} {os.path.basename(directory)} - download complete. {constants.ANSI_RESET}")
        logging.info(f" {os.path.basename(directory)} - download complete.")
    else:
        print(
            f"{constants.ANSI_GREEN} A local instance of {os.path.basename(directory)} has been detected. {constants.ANSI_RESET}")
        logging.info(f" A local instance of {os.path.basename(directory)} has been detected.")
