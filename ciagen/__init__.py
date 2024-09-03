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
        raise OSError(f"Could not add ultralytics path. E: {e}")
    return True


def create_data_folder() -> bool:
    try:
        os.makedirs(REAL_DATAPATH, exist_ok=True)
        os.makedirs(GEN_DATAPATH, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create data folders. E: {e}")
    return True


# def install_pyfeat_if_not_installed() -> bool:
#     # try:
#     #     import feat
#     # except ModuleNotFoundError:
#     import subprocess
#     import sys
#     import site
#     from importlib import reload

#     subprocess.check_call(
#         [
#             sys.executable,
#             "-m",
#             "pip",
#             "install",
#             "-U",
#             "py-feat",
#             "--index-url",
#             "https://github.com/hatellezp/py-feat",
#         ]
#     )

#     reload(site)

#     return True


# Initialize datafolders
create_data_folder()

# Add external libraries's paths
add_ultralytics_path()

# Install pyfeat if not installed
# install_pyfeat_if_not_installed()
