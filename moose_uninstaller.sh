#!/usr/bin/env bash

# **********************************************************************************************************************
# File: moose_uninstaller.sh
# Project: moose-v0.1.0
# Created: 24.06.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: lalith.shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: moose_uninstaller.sh has been particularly created for making the removal process of moose easier
# in linux.
# License: Apache 2.0
# **********************************************************************************************************************

echo "[-] Uninstalling Falcon v0.1.0"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "[2] Linux detected..."
    echo "[3] Removing moose from /usr/local/bin..."
    sudo rm /usr/local/bin/moose
    echo "[4] Removing supporting binaries..."
    sudo rm /usr/local/bin/c3d
    sudo rm /usr/local/bin/greedy
    echo "[5] Removing python dependencies"
    pip uninstall -r requirements.txt
    # shellcheck disable=SC2006
    # shellcheck disable=SC2034
    falcon_dir=`pwd`
    # shellcheck disable=SC2154
    echo "[5] Removing moose folder from $moose_dir..."
    sudo rm -rf "$moose_dir"
    echo "[6] Removing environment variables for moose..."
    # shellcheck disable=SC2154
    unset nnUNet_raw_data_base
    # shellcheck disable=SC2154
    unset nnUNet_preprocessed
    unset RESULTS_FOLDER
    unset IM_SPACE_DIR
    unset BRAIN_DETECTOR_DIR

fi
