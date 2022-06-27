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
    NNUNET_DIR="nnUNet_"
    RESULTS_DIR="RESULTS_FOLDER"
    SIM_SPACE_DIR="SIM_SPACE_DIR"
    BRAIN_DIR="BRAIN_DETECTOR_DIR"
    MOOSE_DIR="MOOSE_DIR"
    BASH_RC_PATH="$HOME/.bashrc"
    sed -i "/${NNUNET_DIR}/d" "${BASH_RC_PATH}"
    sed -i "/${RESULTS_DIR}/d" "${BASH_RC_PATH}"
    sed -i "/${SIM_SPACE_DIR}/d" "${BASH_RC_PATH}"
    sed -i "/${BRAIN_DIR}/d" "${BASH_RC_PATH}"
    sed -i "/${MOOSE_DIR}/d" "${BASH_RC_PATH}"
fi

