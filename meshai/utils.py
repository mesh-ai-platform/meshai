# meshai/utils.py

import os

def create_dir_if_not_exists(directory):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
