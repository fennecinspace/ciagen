import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import DictConfig

from ciagen.data.paths import create_yaml_file
from ciagen.utils.io import create_files_list, list_files


def sort_based_on_score(
    image_paths: List[str], scores: List[float], direction: str = "smaller"
) -> Tuple[List[str], List[int]]:
    combined_data = list(zip(scores, image_paths))
    sorted_data = sorted(
        combined_data,
        key=lambda x: x[0],
        reverse=False if direction == "smaller" else True,
    )
    sorted_scores, sorted_image_paths = zip(*sorted_data)
    return sorted_image_paths, sorted_scores


def mix_yolo(cfg: DictConfig, paths: Dict[str, str | Path]) -> None:

    augmentation_percent = cfg["ml"]["augmentation_percent"]
    train_nb = cfg["ml"]["train_nb"]
    val_nb = cfg["ml"]["val_nb"]
    test_nb = cfg["ml"]["test_nb"]
    _sample = cfg["ml"]["sampling"]
    seed = 42
    formats = cfg["data"]["image_formats"]

    if not os.path.isdir(paths["mixed_yamls_folder_path"]):
        os.makedirs(paths["mixed_yamls_folder_path"])

    train_txt_path = Path(paths["mixed_yamls_folder_path"]) / "train.txt"
    val_txt_path = Path(paths["mixed_yamls_folder_path"]) / "val.txt"
    test_txt_path = Path(paths["mixed_yamls_folder_path"]) / "test.txt"
    data_yaml_path = Path(paths["mixed_yamls_folder_path"]) / "data.yaml"

    real_images_path = Path(paths["real_images"])
    val_images_path = Path(paths["val_images"])
    test_images_path = Path(paths["test_images"])

    real_images = list_files(real_images_path, formats, train_nb)
    val_images = list_files(val_images_path, formats, val_nb)
    test_images = list_files(test_images_path, formats, test_nb)

    synth_images_dir = Path(paths["generated"])

    synth_images = list_files(synth_images_dir, formats)
    random.Random(seed).shuffle(synth_images)

    random.Random(seed).shuffle(real_images)

    nb_synth_images = int(len(real_images) * augmentation_percent)
    synth_images = synth_images[:nb_synth_images]

    train_images = real_images + synth_images

    create_files_list(train_images, train_txt_path)
    create_files_list(val_images, val_txt_path)
    create_files_list(test_images, test_txt_path)

    create_yaml_file(data_yaml_path, train_txt_path, val_txt_path, test_txt_path)

    print(f"Training yaml files created in : {paths['mixed_yamls_folder_path']}")
    print(f"Using {train_nb} Real Images from : ", real_images_path)
    print("Using Synthetic Images from : ", synth_images_dir)
    print(f"Using {val_nb} Validation Images from : ", val_images_path)
    print(f"Using {test_nb} Test Images from : ", test_images_path)

    return data_yaml_path


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from ciagen.data.paths import generate_all_paths

    cfg = OmegaConf.load("ciagen/conf/config.yaml")
    paths = generate_all_paths(cfg)
    mix_yolo(cfg, paths)
