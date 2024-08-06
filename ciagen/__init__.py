import os


REAL_DATAPATH = os.path.join(os.getcwd(), "data", "real")
GEN_DATAPATH = os.path.join(os.getcwd(), "data", "generated")


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
