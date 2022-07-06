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


def dir_exists(dir_path: str) -> bool:
    """
    Checks if a directory exists
    :param dir_path: Path to the directory
    :return: True if directory exists, False otherwise
    """
    return os.path.exists(dir_path)


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
