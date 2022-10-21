#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ***********************************************************************************************************************
# File: checkArgs.py
# Project: MOOSE Version 0.1.0
# Created: 23.03.2022
# Author: Lalith Kumar Shiyam Sundar
# Email: Lalith.Shiyamsundar@meduniwien.ac.at
# Institute: Quantitative Imaging and Medical Physics, Medical University of Vienna
# Description: This module checks if the provided arguments are valid.
# License: Apache 2.0
# **********************************************************************************************************************
import os
import sys


def dir_exists(dir_path: str) -> bool:
    """
    Checks if a directory exists
    :param dir_path: Path to the directory
    :return: True if directory exists, False otherwise
    """
    return os.path.exists(dir_path)


# Function to check if all the environment variables are set
def check_env_vars() -> None:
    """
    Checks if all the environment variables are set
    """
    # Check if all the environment variables are set
    if os.environ.get('RESULTS_FOLDER') is None:
        print("RESULTS_FOLDER environment variable is not set")
        sys.exit("Please set the RESULTS_FOLDER environment variable in .bashrc file")
    elif os.environ.get('nnUNet_preprocessed') is None:
        print("nnUNet_preprocessed environment variable is not set")
        sys.exit("Please set the nnUNet_preprocessed environment variable in .bashrc file")
    elif os.environ.get('nnUNet_raw_data_base') is None:
        print("nnUNet_raw_data_base environment variable is not set")
        sys.exit("Please set the nnUNet_raw_data_base environment variable in .bashrc file")
    elif os.environ.get('SIM_SPACE_DIR') is None:
        print("SIM_SPACE_DIR environment variable is not set")
        sys.exit("Please set the SIM_SPACE_DIR environment variable in .bashrc file")
    elif os.environ.get('BRAIN_DETECTOR_DIR') is None:
        print("BRAIN_DETECTOR_DIR environment variable is not set")
        sys.exit("Please set the BRAIN_DETECTOR_DIR environment variable in .bashrc file")
    else:
        print("All the MOOSE environment variables are set! Proceeding with the execution of the program")



def has_numbers(string: str) -> bool:
    """
    Checks if a string contains number
    :param string: String to check
    :return: True if string contains numbers, False otherwise
    """
    return any(char.isdigit() for char in string)


def is_non_negative(value: int) -> bool:
    """
    Checks if a value is non-negative
    :param value: Value to check
    :return: True if value is non-negative, False otherwise
    """
    return value >= 0


# Function to check if a string has alphabets
def is_string_alpha(string: str) -> bool:
    """
    Checks if a string contains alphabets
    :param string: String to check
    :return: True if string contains alphabets, False otherwise
    """
    return any(char.isalpha() for char in string)


# Function to remove a particular character from a string
def remove_char(string: str, char: str) -> str:
    """
    Removes a particular character from a string
    :param string: String to remove character from
    :param char: Character to remove
    :return: String with the character removed
    """
    return string.replace(char, "")
