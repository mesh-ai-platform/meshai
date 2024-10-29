# meshai/utils.py
import importlib
import os

def create_dir_if_not_exists(directory):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_custom_model(module_path, class_name, *args, **kwargs):
    """
    Dynamically loads a custom model class from a given module path.
    """
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(*args, **kwargs)
