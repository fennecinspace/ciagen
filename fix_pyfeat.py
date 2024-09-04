import sys
import os
import shutil


def get_good_lib_path():
    for p in sys.path:
        if os.path.isdir(p):
            if "feat" in os.listdir(p):
                return p
    return None


if __name__ == "__main__":
    good_lib_path = get_good_lib_path()
    if good_lib_path is None:
        print("Could not find good lib path, are you sure feat is installed")
        sys.exit(1)

    good_stats_path = os.path.join(os.getcwd(), "ciagen", "utils", "stats.py")
    bad_stats_path = os.path.join(good_lib_path, "feat", "utils", "stats.py")
    shutil.copyfile(good_stats_path, bad_stats_path)

    print("copy modified stats.py file, you should be able to import feat now")
