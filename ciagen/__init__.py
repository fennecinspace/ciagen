import os
import sys

PACKAGE_NAME = "ciagen"
ULTRALYTICS_PATH = os.path.join(os.path.join(os.getcwd(), PACKAGE_NAME), "ultralytics")

REAL_DATAPATH = os.path.join(os.getcwd(), "data", "real")
GEN_DATAPATH = os.path.join(os.getcwd(), "data", "generated")


def add_ultralytics_path() -> bool:
    try:
        sys.path.append(ULTRALYTICS_PATH)
    except OSError as e:
        raise Exception(f"Could not add ultralytics path. E: {e}")
    return True


def create_data_folder() -> bool:
    try:
        os.makedirs(REAL_DATAPATH, exist_ok=True)
        os.makedirs(GEN_DATAPATH, exist_ok=True)
    except OSError as e:
        raise Exception(f"Could not create data folders. E: {e}")
    return True


# Initialize datafolders
create_data_folder()


def hello():
    print("Hello, World!")
