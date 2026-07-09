# Copyright 2024 The MathWorks, Inc.

import importlib

def check_library(library_name):
    """
    Check if a library is installed and get its version.

    Args:
        library_name (str): The name of the library to check.

    Returns:
        tuple: A tuple containing a boolean indicating if the library is installed,
               and a string of the version number or a message if not installed.
    """
    try:
        module = importlib.import_module(library_name)
        version = getattr(module, '__version__', 'Version attribute not found')
        return [True, str(version)]
    except ImportError:
        return [False, 'Not Installed']