#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

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
from tqdm import tqdm
from moosez import constants
from moosez import resources


def binary(system_info, url):
    """
    Downloads the binary for the current system.
    :param system_info: A dictionary containing the system information.
    :param url: The url to download the binary from.
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
    :param model_path: The path to store the model.
    """
    model_info = resources.MODELS[model_name]
    url = model_info["url"]
    filename = os.path.join(model_path, model_info["filename"])
    directory = os.path.join(model_path, model_info["directory"])

    if not os.path.exists(directory):
        logging.info(f" Downloading {directory}")
        # show progress using tqdm
        with tqdm(unit="B", unit_scale=True, leave=False, desc=f" Downloading {os.path.basename(directory)}") as pbar:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("Content-Length", 0))
            pbar.total = total_size
            chunk_size = 1024 * 10
            for chunk in response.iter_content(chunk_size=chunk_size):
                open(filename, "ab").write(chunk)
                pbar.update(chunk_size)
        # Unzip the model
        # Unzip the model
        import zipfile
        with tqdm(unit="B", unit_scale=True, leave=False, desc=f" Extracting {os.path.basename(directory)}") as pbar:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                total_size = sum((file.file_size for file in zip_ref.infolist()))
                pbar.total = total_size
                # Get the parent directory of 'directory'
                parent_directory = os.path.dirname(directory)
                for file in zip_ref.infolist():
                    zip_ref.extract(file, parent_directory)
                    extracted_size = file.file_size
                    pbar.update(extracted_size)
        logging.info(f" {os.path.basename(directory)} extracted.")

        # Delete the zip file
        os.remove(filename)
        print(f"{constants.ANSI_GREEN} {os.path.basename(directory)} - download complete. {constants.ANSI_RESET}")
        logging.info(f" {os.path.basename(directory)} - download complete.")
    else:
        print(f"{constants.ANSI_GREEN} A local instance of {os.path.basename(directory)} has been detected. "
              f"{constants.ANSI_RESET}")
        logging.info(f" A local instance of {os.path.basename(directory)} has been detected.")
