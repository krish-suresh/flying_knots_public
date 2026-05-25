# TODO change this file name to be more meaningful
import yaml
import os
import pickle
from datetime import datetime
import hashlib
import platform

def get_platform():
    system = platform.system()

    if system == "Darwin":
        return "mac"
    elif system == "Linux":
        return "ubuntu"
    else:
        return system

def get_flying_knot_data_dir():
    return os.path.expanduser(os.environ.get("FLYING_KNOT_DATA", "~/flying_knot_data"))

def dict_to_yaml(file_path, data):
    def represent_inline_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, represent_inline_list)

    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)   


def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_pickle(file_path):
    with open(file_path, "rb") as openfile:
        return pickle.load(openfile)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_latest_trial_name(folder):
    subfolders = [
        f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))
    ]

    def extract_datetime(name: str):
        try:
            return datetime.strptime(
                name.split("-")[0] + name.split("-")[1], "%Y%m%d%H%M%S"
            )
        except (ValueError, IndexError):
            return None

    dated_folders = [(f, extract_datetime(f)) for f in subfolders]
    dated_folders = [(f, d) for f, d in dated_folders if d is not None]

    if not dated_folders:
        raise ValueError("No valid trial folders found.")

    latest_folder = max(dated_folders, key=lambda x: x[1])[0]
    return latest_folder


def hash_dict(d):
    hash = hashlib.sha256(pickle.dumps(d))
    return hash.hexdigest()
