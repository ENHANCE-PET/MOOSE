#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ***********************************************************************************************************************
# File: fileOp.py
# Project: MOOSE Version 1.0
# Created: 21.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: Lalith.Shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This module contains the functions to perform basic file and folder operations
# License: Apache 2.0
# **********************************************************************************************************************

import glob
import json
import os
import pathlib
import re
import shutil
from pathlib import Path
import natsort
import pyfiglet

def display_logo():
    """
    Display MOOSE logo
    :return:
    """
    print("\n")
    result = pyfiglet.figlet_format("MOOSE v0.1", font="slant")
    print(result)


def display_citation():
    """
    Display manuscript citation
    :return:
    """
    print(" L. K. Shiyam Sundar et al. “Fully-automated, semantic segmentation of whole-body [18F]FDG PET/CT images based on data-centric artificial intelligence”. Accepted: Journal of Nuclear Medicine (2022).")
    print(" Copyright 2022, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")

def get_folders(dir_path: str) -> list:
    """
    Returns a list of all folders in a directory
    :param dir_path: Main directory containing all folders
    :return: List of folder paths
    """
    # Get a list of folders inside a directory using glob
    folders = glob.glob(os.path.join(dir_path, "*"))
    # Sort the list of folders by name
    folders = natsort.natsorted(folders)
    return folders


def get_files(dir_path: str, wildcard: str) -> list:
    """
    Returns a list of all files in a directory
    :param dir_path: Folder containing all files
    :param wildcard: Wildcard to filter files
    :return: List of file paths
    """
    # Get a list of files inside a directory using glob
    files = glob.glob(os.path.join(dir_path, wildcard))
    # Sort the list of files by name
    files = natsort.natsorted(files)
    return files


def make_dir(dir_path: str, dir_name: str) -> str:
    """
    Creates a new directory
    :param dir_path: Directory path to create the new directory in
    :param dir_name: Name of the new directory
    :return: Path to the new directory
    """
    # Create a directory with user specified name if it does not exist
    if not os.path.exists(os.path.join(dir_path, dir_name)):
        os.mkdir(os.path.join(dir_path, dir_name))

    return os.path.join(dir_path, dir_name)


def move_files(src_dir: str, dest_dir: str, wildcard: str) -> None:
    """
    Moves files from one directory to another
    :param src_dir: Source directory from which files are moved
    :param dest_dir: Target directory to which files are moved
    :param wildcard: Wildcard to filter files that are moved
    :return: None
    """
    # Get a list of files using wildcard
    files = get_files(src_dir, wildcard)
    # Move each file from source directory to destination directory
    for file in files:
        os.rename(file, os.path.join(dest_dir, os.path.basename(file)))


def copy_files(src_dir: str, dest_dir: str, wildcard: str) -> None:
    """
    Copies files from one directory to another
    :param src_dir: Source directory from which files are copied
    :param dest_dir: Target directory to which files are copied
    :param wildcard: Wildcard to filter files that are copied
    :return: None
    """
    # Get a list of files using wildcard
    files = get_files(src_dir, wildcard)
    # Copy each file from source directory to destination directory
    for file in files:
        shutil.copy(file, dest_dir)


def delete_files(dir_path: str, wildcard: str) -> None:
    """
    Deletes files from a directory
    :param dir_path: Path to the directory from which files are deleted
    :param wildcard: Wildcard to filter files that are deleted
    :return: None
    """
    # Get a list of files using wildcard
    files = get_files(dir_path, wildcard)
    # Delete each file from directory
    for file in files:
        os.remove(file)


def compress_file(file: str) -> str:
    """
    Compresses files using pigz
    :param file: File to be compressed
    :return: None
    """
    os.system("pigz " + re.escape(file))
    return file + ".gz"


def read_json(file_path: str) -> dict:
    """
    Reads a json file and returns a dictionary
    :param file_path: Path to the json file
    :return: Dictionary
    """
    # Open the json file
    with open(file_path, "r") as json_file:
        # Read the json file and return the dictionary
        return json.load(json_file)


def organise_nii_files_in_folders(dir_path: str, json_files: list) -> None:
    """
    Organises the nifti files in their respective 'modality' folders
    :param dir_path: Directory containing nifti files
    :param json_files: Path to the JSON file
    :return: None
    """
    os.chdir(dir_path)
    for json_file in json_files:
        nifti_file = Path(json_file).stem + ".nii"
        if os.path.exists(nifti_file):
            # Get the modality from the json file
            modality = read_json(json_file)["Modality"]
            # Create a new directory for the modality if it does not exist
            make_dir(dir_path, modality)
            # Move the nifti file to the new directory
            move_files(src_dir=dir_path, dest_dir=os.path.join(dir_path, modality), wildcard=nifti_file)


def add_suffix_rename(file_path: str, suffix: str) -> str:
    """
    Adds a suffix to the file name and renames the file
    :param file_path: Path to the file
    :param suffix: Suffix to be added
    :return: Path to the file with the suffix
    """
    p = pathlib.Path(file_path)
    p.rename(pathlib.Path(p.parent, "{}_{}{}".format(p.stem, suffix, p.suffix)))
    return str(pathlib.Path(p.parent, "{}_{}{}".format(p.stem, suffix, p.suffix)))


def add_prefix_rename(file_path: str, prefix: str) -> str:
    """
    Adds a prefix to the file name
    :param file_path: Path to the file
    :param prefix: Prefix to be added
    :return: Path to the file with the prefix
    """
    p = pathlib.Path(file_path)
    p.rename(pathlib.Path(p.parent, "{}_{}{}".format(prefix, p.stem, p.suffix)))
    return str(pathlib.Path(p.parent, "{}_{}{}".format(prefix, p.stem, p.suffix)))


def add_suffix(file_path: str, suffix: str) -> str:
    """
    Adds a suffix to the file name
    :param file_path: Path to the file
    :param suffix: Suffix to be added
    :return: Path to the file with the suffix
    """
    return str(pathlib.Path(file_path).with_suffix("{}{}".format(suffix, pathlib.Path(file_path).suffix)))


def add_prefix(file_path: str, prefix: str) -> str:
    """
    Adds a prefix to the file name
    :param file_path: Path to the file
    :param prefix: Prefix to be added
    :return: Path to the file with the prefix
    """
    return str(pathlib.Path(file_path).with_name("{}{}".format(prefix, pathlib.Path(file_path).name)))


def move_contents(content_list: list, dest_dir: str) -> None:
    """
    Moves contents of a directory to another directory
    :param content_list: List of contents to be moved
    :param dest_dir: Destination directory
    :return: None
    """
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    for content in content_list:
        if os.path.exists(content) and os.path.isdir(content):
            shutil.move(content, dest_dir)
        elif os.path.exists(content) and os.path.isfile(content):
            shutil.move(content, dest_dir)
