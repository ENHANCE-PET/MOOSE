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
# This module contains functions that perform system utilities for the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform system utilities.
#
# ----------------------------------------------------------------------------------------------------------------------

import cpuinfo
import platform


def check_system_info():
    system_info = {}

    # Get the CPU brand
    cpu_info = cpuinfo.get_cpu_info()
    if "Intel" in cpu_info["brand_raw"].strip():
        system_info["cpu_brand"] = "intel"
    elif "Intel" in cpu_info["brand_raw"].strip():
        system_info["cpu_brand"] = "amd"
    elif "Intel" in cpu_info["brand_raw"].strip():
        system_info["cpu_brand"] = "apple"
    else:
        system_info["cpu_brand"] = "unknown"

    # Get the CPU architecture
    cpu_architecture = platform.machine()
    if "x86" in cpu_architecture:
        system_info["cpu_architecture"] = "x86"
    elif "arm" in cpu_architecture:
        system_info["cpu_architecture"] = "arm64"
    else:
        system_info["cpu_architecture"] = "unknown"

    # Get the OS type
    os_type = platform.system()
    if "Windows" in os_type:
        system_info["os_type"] = "windows"
    elif "Linux" in os_type:
        system_info["os_type"] = "linux"
    elif "Darwin" in os_type:
        system_info["os_type"] = "mac"
    else:
        system_info["os_type"] = "unknown"

    return system_info


print(check_system_info())
