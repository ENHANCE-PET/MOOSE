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
# This module contains the functions for performing file operations for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform file operations.
#
# ----------------------------------------------------------------------------------------------------------------------

import os


def create_directory(directory_path: str):
    """
    Creates a directory at the specified path.
    :param directory_path: The path to the directory.
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
