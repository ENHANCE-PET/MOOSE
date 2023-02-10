#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 09.02.2023
# Version: 2.0.0
#
# Description:
# This module performs input validation for the moosez. It verifies that the inputs provided by the user are valid
# and meets the required specifications.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform input validation.
#
# ----------------------------------------------------------------------------------------------------------------------

import os
import sys
import argparse
import logging


def check_directory_exists(directory_path: str):
    """
    Checks if the specified directory exists.
    :param directory_path: The path to the directory.
    :raises: Exception if the directory does not exist.
    """
    if not os.path.isdir(directory_path):
        raise Exception(f"Error: The directory '{directory_path}' does not exist.")



