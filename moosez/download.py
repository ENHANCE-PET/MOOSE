#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import os
import resources


def binary(system_info, url):
    binary_name = "{}_{}_{}".format(system_info["os_type"], system_info["cpu_architecture"], system_info["cpu_brand"])
    print("Binary to download: " + binary_name)
    response = requests.get(url + binary_name)

    with open(binary_name, "wb") as f:
        f.write(response.content)


def model(model_name):
    model_info = resources.MODELS[model_name]
    url = model_info["url"]
    filename = model_info["filename"]
    directory = model_info["directory"]

    if not os.path.exists(directory):
        print(f"Downloading {directory}...")
        response = requests.get(url)
        open(filename, "wb").write(response.content)

        # Unzip the model
        import zipfile
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(directory)

        print(f"{directory} downloaded and extracted.")
    else:
        print(f"{directory} already exists.")
